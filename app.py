"""
LLM Safety Gateway API
CPU-centric, lightweight real-time safety gateway for LLM prompts.

Runs 4 detection layers in parallel and uses weighted scoring to determine
whether to PASS, SANITIZE, or BLOCK prompts.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import all layer classes
from utils.reg import SemanticSanitizer
from utils.security_model import SecurityClassifier
from utils.mutation_layer import PromptMutator
from utils.extractor import BottleneckExtractor
from utils.sanitizer import PromptSanitizer


# =============================================================================
# CONFIGURATION
# =============================================================================

class GatewayConfig:
    """Configuration for the safety gateway."""
    
    # Layer Weights (must sum to 1.0)
    WEIGHT_REGEX = 0.20          # Fast keyword/pattern detection
    WEIGHT_SECURITY_MODEL = 0.35  # Trained classifier (most reliable)
    WEIGHT_MUTATION = 0.3        # Adversarial robustness testing
    WEIGHT_BOTTLENECK = 0.15      # Semantic similarity to jailbreaks
    
    # Decision Thresholds
    THRESHOLD_PASS = 0.4          # Below this: PASS (safe)
    THRESHOLD_SANITIZE = 0.7      # Below this: SANITIZE, Above: BLOCK


class Action(str, Enum):
    """Possible actions for a prompt."""
    PASS = "pass"
    SANITIZE = "sanitize"
    BLOCK = "block"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PromptRequest(BaseModel):
    """Request model for prompt analysis."""
    prompt: str = Field(..., min_length=1, description="The prompt to analyze")


class LayerScore(BaseModel):
    """Score from a single layer."""
    name: str
    score: float
    weight: float
    weighted_score: float


class AnalysisResponse(BaseModel):
    """Response model for prompt analysis."""
    prompt_preview: str
    action: Action
    weighted_score: float
    layer_scores: list[LayerScore]
    sanitized_prompt: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    layers_loaded: dict[str, bool]


# =============================================================================
# APPLICATION LIFESPAN (Model Initialization)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all models at startup, clean up on shutdown.
    This ensures models are loaded ONCE when the server starts.
    """
    print("\n" + "="*60)
    print("🚀 LLM SAFETY GATEWAY - INITIALIZING")
    print("="*60 + "\n")
    
    # Store layer instances in app state
    try:
        # Layer 1: Regex Analyzer
        print("📦 Loading Layer 1: Regex Analyzer...")
        app.state.regex_layer = SemanticSanitizer()
        
        # Layer 2: Security Model (Transformer Classifier)
        print("\n📦 Loading Layer 2: Security Model...")
        app.state.security_model = SecurityClassifier()
        
        # Layer 3: Mutation Layer (shares security model to avoid double-loading)
        print("\n📦 Loading Layer 3: Mutation Layer...")
        app.state.mutation_layer = PromptMutator(classifier=app.state.security_model)
        
        # Layer 4: Bottleneck Extractor (ChromaDB)
        print("\n📦 Loading Layer 4: Bottleneck Extractor...")
        app.state.bottleneck_layer = BottleneckExtractor()
        
        # Sanitizer Layer (only used when needed)
        print("\n📦 Loading Sanitizer Layer...")
        try:
            app.state.sanitizer = PromptSanitizer()
            app.state.sanitizer_available = True
        except ValueError as e:
            print(f"[!] Warning: {e}")
            print("[!] Sanitizer will not be available - prompts requiring sanitization will be blocked.")
            app.state.sanitizer = None
            app.state.sanitizer_available = False
        
        print("\n" + "="*60)
        print("✅ ALL LAYERS LOADED SUCCESSFULLY")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ FATAL: Failed to initialize layers: {e}")
        raise
    
    yield  # Server runs here
    
    # Cleanup (if needed)
    print("\n🛑 Shutting down LLM Safety Gateway...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="LLM Safety Gateway",
    description="CPU-centric, lightweight real-time safety gateway for LLM prompts",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def run_all_layers(prompt: str, app_state) -> list[LayerScore]:
    """
    Run all 4 detection layers in PARALLEL and return their scores.
    """
    
    async def run_regex():
        score = await app_state.regex_layer.get_score_async(prompt)
        return LayerScore(
            name="regex_analyzer",
            score=round(score, 4),
            weight=GatewayConfig.WEIGHT_REGEX,
            weighted_score=round(score * GatewayConfig.WEIGHT_REGEX, 4)
        )
    
    async def run_security_model():
        result = await app_state.security_model.get_score_async(prompt)
        # Convert to risk score: dangerous = confidence, safe = 1 - confidence
        if result['is_dangerous']:
            score = result['confidence']
        else:
            score = 1.0 - result['confidence']
        return LayerScore(
            name="security_model",
            score=round(score, 4),
            weight=GatewayConfig.WEIGHT_SECURITY_MODEL,
            weighted_score=round(score * GatewayConfig.WEIGHT_SECURITY_MODEL, 4)
        )
    
    async def run_mutation():
        score = await app_state.mutation_layer.get_score_async(prompt)
        return LayerScore(
            name="mutation_layer",
            score=round(score, 4),
            weight=GatewayConfig.WEIGHT_MUTATION,
            weighted_score=round(score * GatewayConfig.WEIGHT_MUTATION, 4)
        )
    
    async def run_bottleneck():
        score = await app_state.bottleneck_layer.get_score_async(prompt)
        return LayerScore(
            name="bottleneck_extractor",
            score=round(score, 4),
            weight=GatewayConfig.WEIGHT_BOTTLENECK,
            weighted_score=round(score * GatewayConfig.WEIGHT_BOTTLENECK, 4)
        )
    
    # Run all layers in parallel
    results = await asyncio.gather(
        run_regex(),
        run_security_model(),
        run_mutation(),
        run_bottleneck()
    )
    
    return list(results)


def determine_action(weighted_score: float) -> Action:
    """Determine action based on weighted score and thresholds."""
    if weighted_score < GatewayConfig.THRESHOLD_PASS:
        return Action.PASS
    elif weighted_score < GatewayConfig.THRESHOLD_SANITIZE:
        return Action.SANITIZE
    else:
        return Action.BLOCK


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if all layers are loaded and ready."""
    return HealthResponse(
        status="healthy",
        layers_loaded={
            "regex_analyzer": hasattr(app.state, 'regex_layer'),
            "security_model": hasattr(app.state, 'security_model'),
            "mutation_layer": hasattr(app.state, 'mutation_layer'),
            "bottleneck_extractor": hasattr(app.state, 'bottleneck_layer'),
            "sanitizer": getattr(app.state, 'sanitizer_available', False)
        }
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_prompt(request: PromptRequest):
    """
    Analyze a prompt through all safety layers WITHOUT sanitizing.
    Returns the action (pass/sanitize/block) and detailed scores.
    """
    prompt = request.prompt
    
    # Run all layers in parallel
    layer_scores = await run_all_layers(prompt, app.state)
    
    # Calculate weighted score
    weighted_score = sum(layer.weighted_score for layer in layer_scores)
    
    # Determine action
    action = determine_action(weighted_score)
    
    return AnalysisResponse(
        prompt_preview=prompt[:512] + "..." if len(prompt) > 512 else prompt,
        action=action,
        weighted_score=round(weighted_score, 4),
        layer_scores=layer_scores,
        sanitized_prompt=None
    )


@app.post("/gateway", response_model=AnalysisResponse)
async def gateway(request: PromptRequest):
    """
    Full safety gateway pipeline:
    1. Analyze prompt through all layers (in parallel)
    2. Calculate weighted score
    3. PASS if safe, BLOCK if dangerous, SANITIZE if in between
    
    This is the main endpoint for protecting LLM applications.
    """
    prompt = request.prompt
    
    # Run all layers in parallel
    layer_scores = await run_all_layers(prompt, app.state)
    
    # Calculate weighted score
    weighted_score = sum(layer.weighted_score for layer in layer_scores)
    
    # Determine action
    action = determine_action(weighted_score)
    
    # Handle sanitization if needed
    sanitized_prompt = None
    if action == Action.SANITIZE:
        if app.state.sanitizer_available:
            try:
                sanitized_prompt = await app.state.sanitizer.sanitize_async(prompt)
            except Exception as e:
                print(f"[!] Sanitization failed: {e}")
                # Fall back to blocking if sanitization fails
                action = Action.BLOCK
        else:
            # No sanitizer available, block instead
            action = Action.BLOCK
    
    return AnalysisResponse(
        prompt_preview=prompt[:512] + "..." if len(prompt) > 512 else prompt,
        action=action,
        weighted_score=round(weighted_score, 4),
        layer_scores=layer_scores,
        sanitized_prompt=sanitized_prompt
    )


@app.get("/config")
async def get_config():
    """Get current gateway configuration (weights and thresholds)."""
    return {
        "weights": {
            "regex_analyzer": GatewayConfig.WEIGHT_REGEX,
            "security_model": GatewayConfig.WEIGHT_SECURITY_MODEL,
            "mutation_layer": GatewayConfig.WEIGHT_MUTATION,
            "bottleneck_extractor": GatewayConfig.WEIGHT_BOTTLENECK
        },
        "thresholds": {
            "pass": GatewayConfig.THRESHOLD_PASS,
            "sanitize": GatewayConfig.THRESHOLD_SANITIZE
        },
        "actions": {
            "score < threshold_pass": "PASS (safe, forward to LLM)",
            "threshold_pass <= score < threshold_sanitize": "SANITIZE (clean before forwarding)",
            "score >= threshold_sanitize": "BLOCK (reject entirely)"
        }
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
