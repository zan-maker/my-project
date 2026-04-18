"""
MineLens AI - Critical Mineral Prospectivity Mapping with Gemma 4
Backend API Server (FastAPI)

Gemma 4 Good Hackathon: https://www.kaggle.com/competitions/gemma-4-good-hackathon
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json
import os

# ============================================================
# Function Calling Tool Definitions
# ============================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "spectral_analysis",
            "description": "Analyze satellite imagery to identify mineral spectral signatures. Detects anomalies in reflectance patterns that indicate presence of critical minerals like lithium, cobalt, rare earths, copper, nickel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the satellite image file"},
                    "mineral_targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of target minerals to detect (e.g., ['lithium', 'cobalt', 'rare_earth'])"
                    },
                    "confidence_threshold": {"type": "number", "description": "Minimum confidence score (0-1)", "default": 0.6}
                },
                "required": ["image_path", "mineral_targets"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terrain_classifier",
            "description": "Classify terrain types from satellite imagery or elevation data. Identifies geological formations associated with mineral deposits (e.g., pegmatites, greenstone belts, porphyry intrusions).",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the satellite or DEM image"},
                    "classification_detail": {"type": "string", "enum": ["basic", "detailed", "expert"], "default": "detailed"}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "proximity_search",
            "description": "Search for known mineral deposits, mines, and geological features near a given location. Uses USGS MRDS and other public databases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Latitude of the center point"},
                    "longitude": {"type": "number", "description": "Longitude of the center point"},
                    "radius_km": {"type": "number", "description": "Search radius in kilometers", "default": 50},
                    "mineral_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by mineral types (e.g., ['copper', 'gold', 'lithium'])"
                    }
                },
                "required": ["latitude", "longitude"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "risk_assessment",
            "description": "Assess geopolitical and supply chain risks for critical mineral operations in a region. Considers trade policies, environmental regulations, political stability, and infrastructure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "description": "Region or country name"},
                    "mineral_type": {"type": "string", "description": "The mineral of interest"},
                    "factors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Risk factors to assess: ['political', 'environmental', 'infrastructure', 'trade', 'social']",
                        "default": ["political", "environmental", "infrastructure", "trade"]
                    }
                },
                "required": ["region", "mineral_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a comprehensive mineral prospectivity report based on analysis results. Includes executive summary, methodology, findings, confidence scores, and recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location being analyzed"},
                    "mineral_targets": {"type": "array", "items": {"type": "string"}, "description": "Target minerals"},
                    "analysis_results": {"type": "object", "description": "Combined results from other analysis tools"},
                    "report_type": {"type": "string", "enum": ["executive", "detailed", "technical"], "default": "detailed"}
                },
                "required": ["location", "mineral_targets", "analysis_results"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "geological_survey_lookup",
            "description": "Look up geological survey data for a region. Retrieves information about rock types, formations, fault lines, and known mineral occurrences from USGS and state geological surveys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Latitude"},
                    "longitude": {"type": "number", "description": "Longitude"},
                    "data_layers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Data layers to retrieve: ['geology', 'mineral_occurrences', 'fault_lines', 'geochemistry', 'magnetics', 'gravity']",
                        "default": ["geology", "mineral_occurrences"]
                    }
                },
                "required": ["latitude", "longitude"]
            }
        }
    }
]


# ============================================================
# Tool Implementations
# ============================================================

class ToolRegistry:
    """Registry and execution engine for function calling tools."""
    
    def __init__(self):
        self.tools = {}
        self._register_defaults()
    
    def _register_defaults(self):
        from tools.spectral import spectral_analysis
        from tools.terrain import terrain_classifier
        from tools.proximity import proximity_search
        from tools.risk import risk_assessment
        from tools.report import generate_report
        
        self.tools = {
            "spectral_analysis": spectral_analysis,
            "terrain_classifier": terrain_classifier,
            "proximity_search": proximity_search,
            "risk_assessment": risk_assessment,
            "generate_report": generate_report,
        }
    
    def call(self, function_name: str, arguments: Dict) -> Dict:
        """Execute a tool call."""
        tool = self.tools.get(function_name)
        if not tool:
            raise ValueError(f"Unknown tool: {function_name}")
        return tool(**arguments)
    
    def get_schema(self) -> List[Dict]:
        """Get the tools schema for Gemma function calling."""
        return TOOLS_SCHEMA


# ============================================================
# Gemma 4 Client with Function Calling
# ============================================================

class GemmaClient:
    """Client for Gemma 4 with function calling support."""
    
    def __init__(self, model_path: str = "google/gemma-4-E2B-it"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
    
    def load_model(self):
        """Load Gemma 4 model. For Kaggle, model weights are pre-available."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        print(f"Loading Gemma 4 from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        except Exception:
            self.processor = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("Gemma 4 loaded successfully!")
    
    def chat(self, message: str, image: Optional[Any] = None) -> str:
        """Send a message and get response, with automatic function calling."""
        
        # Build conversation
        user_msg = {"role": "user", "content": message}
        if image is not None:
            user_msg["content"] = [
                {"type": "image", "image": image},
                {"type": "text", "text": message}
            ]
        
        self.conversation_history.append(user_msg)
        
        # Generate response
        response = self._generate()
        
        # Handle function calls
        max_tool_rounds = 5
        tool_round = 0
        
        while self._has_tool_calls(response) and tool_round < max_tool_rounds:
            self.conversation_history.append({"role": "assistant", "content": response})
            
            tool_calls = self._extract_tool_calls(response)
            for call in tool_calls:
                func_name = call["name"]
                func_args = call["arguments"]
                
                print(f"  Tool call: {func_name}({func_args})")
                
                # Execute tool
                try:
                    result = self.tool_registry.call(func_name, func_args)
                    tool_response = json.dumps(result, indent=2)
                except Exception as e:
                    tool_response = json.dumps({"error": str(e)})
                
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Tool result for {func_name}:\n{tool_response}"
                })
            
            response = self._generate()
            tool_round += 1
        
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def _generate(self) -> str:
        """Generate response from conversation history."""
        if self.processor and any(
            isinstance(msg.get("content"), list) 
            for msg in self.conversation_history 
            if isinstance(msg.get("content"), list)
        ):
            # Multimodal input
            inputs = self.processor.apply_chat_template(
                self.conversation_history,
                tools=self.tool_registry.get_schema(),
                return_tensors="pt",
                add_generation_prompt=True,
            )
        else:
            # Text only
            inputs = self.tokenizer.apply_chat_template(
                self.conversation_history,
                tools=self.tool_registry.get_schema(),
                return_tensors="pt",
                add_generation_prompt=True,
            )
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response
    
    def _has_tool_calls(self, response: str) -> bool:
        """Check if response contains tool calls."""
        return "<tool_call)" in response or "function_call" in response.lower()
    
    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """Extract function calls from model response."""
        calls = []
        # Parse function call format from Gemma
        import re
        
        # Match patterns like: function_name({"param": "value"})
        pattern = r'(\w+)\s*\(\s*(\{[^}]*\}|"[^"]*")\s*\)'
        for match in re.finditer(pattern, response):
            func_name = match.group(1)
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}
            calls.append({"name": func_name, "arguments": args})
        
        return calls
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def analyze_mineral_site(
        self, 
        latitude: float, 
        longitude: float, 
        mineral_targets: List[str],
        image_path: Optional[str] = None,
    ) -> Dict:
        """Full agentic analysis pipeline for a mineral site."""
        
        self.reset_conversation()
        
        prompt = f"""Analyze this location for critical mineral prospectivity:

Location: ({latitude}, {longitude})
Target minerals: {', '.join(mineral_targets)}

Please perform a comprehensive analysis:
1. Use geological_survey_lookup to get geological context
2. Use proximity_search to find nearby deposits
3. Use risk_assessment to evaluate supply chain risks
4. Use spectral_analysis if an image is available
5. Use generate_report to compile findings

Provide a detailed prospectivity assessment with confidence scores."""
        
        if image_path:
            response = self.chat(prompt, image=image_path)
        else:
            response = self.chat(prompt)
        
        return {
            "location": {"lat": latitude, "lon": longitude},
            "mineral_targets": mineral_targets,
            "analysis": response,
            "conversation_length": len(self.conversation_history),
        }


# ============================================================
# API Models
# ============================================================

class AnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    mineral_targets: List[str] = ["lithium", "cobalt", "rare_earth", "copper", "nickel"]
    report_type: str = "detailed"

class ChatRequest(BaseModel):
    message: str

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class ProspectsNearMeRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 100
    mineral_type: Optional[str] = None


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="MineLens AI",
    description="Critical Mineral Prospectivity Mapping powered by Gemma 4",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global client
gemma_client: Optional[GemmaClient] = None

@app.on_event("startup")
async def startup():
    global gemma_client
    gemma_client = GemmaClient(
        model_path=os.environ.get("GEMMA_MODEL_PATH", "google/gemma-4-E2B-it")
    )
    try:
        gemma_client.load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Running in demo mode without model inference.")


@app.get("/")
async def root():
    return {
        "service": "MineLens AI",
        "description": "Critical Mineral Prospectivity Mapping with Gemma 4",
        "endpoints": {
            "/analyze": "POST - Full mineral site analysis",
            "/chat": "POST - Chat with the AI geologist",
            "/tool_call": "POST - Execute a specific tool",
            "/prospects_near": "POST - Find prospects near location",
            "/upload": "POST - Upload satellite imagery",
            "/tools": "GET - List available tools",
            "/health": "GET - Health check",
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": gemma_client is not None and gemma_client.model is not None,
    }


@app.get("/tools")
async def list_tools():
    return {"tools": TOOLS_SCHEMA}


@app.post("/analyze")
async def analyze_site(request: AnalysisRequest):
    """Full agentic analysis of a mineral site."""
    if not gemma_client or not gemma_client.model:
        raise HTTPException(503, "Model not loaded. Running in demo mode.")
    
    result = gemma_client.analyze_mineral_site(
        latitude=request.latitude,
        longitude=request.longitude,
        mineral_targets=request.mineral_targets,
    )
    return result


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the AI geologist."""
    if not gemma_client or not gemma_client.model:
        return {"response": "Demo mode: Model not loaded. Please run on Kaggle for full functionality."}
    
    response = gemma_client.chat(request.message)
    return {"response": response}


@app.post("/tool_call")
async def execute_tool(request: ToolCallRequest):
    """Execute a specific tool directly."""
    registry = ToolRegistry()
    try:
        result = registry.call(request.tool_name, request.arguments)
        return {"tool": request.tool_name, "result": result}
    except Exception as e:
        raise HTTPException(400, f"Tool error: {str(e)}")


@app.post("/prospects_near")
async def prospects_near(request: ProspectsNearMeRequest):
    """Find mineral prospects near a location."""
    registry = ToolRegistry()
    result = registry.call("proximity_search", {
        "latitude": request.latitude,
        "longitude": request.longitude,
        "radius_km": request.radius_km,
        "mineral_types": [request.mineral_type] if request.mineral_type else None,
    })
    return result


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload satellite imagery for analysis."""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    return {
        "filename": file.filename,
        "path": file_path,
        "size": os.path.getsize(file_path),
        "message": "Image uploaded. Use /analyze with image_path to run analysis."
    }


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting MineLens AI server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
