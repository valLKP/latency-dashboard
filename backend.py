"""
Linkup Latency Test Dashboard
FastAPI server for running latency distribution tests on Linkup API.
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import time
import statistics
from typing import Literal
import os

app = FastAPI(title="Linkup Latency Dashboard")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION
# ============================================================

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY", "6838a568-4275-4b0e-b84e-457bb4d337f1")
LINKUP_ENDPOINT = "https://api.linkup.so/v1/search"

TEST_QUERIES = [
    "What is Microsoft's 2024 revenue?",
    "What is the latest news in AI?",
    "Current technology trends in 2024",
    "OpenAI GPT-4 capabilities",
    "Climate change latest research",
    "Tesla stock performance 2024",
    "Apple Vision Pro reviews",
    "SpaceX Starship updates",
    "Bitcoin price prediction",
    "Best programming languages 2024",
]

# Store for test results
test_results = {
    "status": "idle",
    "progress": 0,
    "total_requests": 0,
    "completed_requests": 0,
    "results": [],
    "stats": None,
    "distribution": None,
}


class TestConfig(BaseModel):
    num_requests: int = 100
    depth: Literal["fast", "deep"] = "fast"


def test_linkup(query: str, depth: str = "fast") -> dict:
    """Make a single request to Linkup API."""
    headers = {
        "Authorization": f"Bearer {LINKUP_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "q": query,
        "depth": depth,
        "outputType": "searchResults",
        "includeSources": False,
        "includeImages": False,
    }
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(LINKUP_ENDPOINT, json=payload, headers=headers, timeout=60)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "query": query[:40],
            "latency_ms": round(latency_ms, 2),
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }
    except Exception as e:
        end_time = time.perf_counter()
        return {
            "query": query[:40],
            "latency_ms": round((end_time - start_time) * 1000, 2),
            "status_code": 0,
            "success": False,
        }


def calculate_stats(results: list) -> dict:
    """Calculate statistics from results."""
    successful = [r for r in results if r["success"]]
    latencies = [r["latency_ms"] for r in successful]
    
    if not latencies:
        return None
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "total": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "success_rate": round((len(successful) / len(results)) * 100, 1),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "avg_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
        "p50_ms": round(sorted_latencies[int(n * 0.50)], 2),
        "p90_ms": round(sorted_latencies[int(n * 0.90)], 2) if n >= 10 else round(sorted_latencies[-1], 2),
        "p95_ms": round(sorted_latencies[int(n * 0.95)], 2) if n >= 20 else round(sorted_latencies[-1], 2),
        "p99_ms": round(sorted_latencies[int(n * 0.99)], 2) if n >= 100 else round(sorted_latencies[-1], 2),
    }


def calculate_distribution(results: list, num_bins: int = 20) -> dict:
    """Calculate latency distribution for histogram."""
    successful = [r for r in results if r["success"]]
    latencies = [r["latency_ms"] for r in successful]
    
    if not latencies:
        return None
    
    min_lat = min(latencies)
    max_lat = max(latencies)
    
    # Create bins
    bin_width = (max_lat - min_lat) / num_bins if max_lat > min_lat else 1
    bins = []
    counts = []
    
    for i in range(num_bins):
        bin_start = min_lat + (i * bin_width)
        bin_end = min_lat + ((i + 1) * bin_width)
        bin_label = f"{int(bin_start)}-{int(bin_end)}"
        
        count = sum(1 for lat in latencies if bin_start <= lat < bin_end)
        if i == num_bins - 1:  # Include max in last bin
            count = sum(1 for lat in latencies if bin_start <= lat <= bin_end)
        
        bins.append(bin_label)
        counts.append(count)
    
    return {
        "bins": bins,
        "counts": counts,
        "bin_width": round(bin_width, 2),
        "latencies": latencies,  # Raw data for scatter/line chart
    }


@app.get("/")
async def serve_frontend():
    """Serve the frontend."""
    return FileResponse("index.html")


@app.get("/api/status")
async def get_status():
    """Get current test status and results."""
    return {
        "status": test_results["status"],
        "progress": test_results["progress"],
        "total_requests": test_results["total_requests"],
        "completed_requests": test_results["completed_requests"],
        "results": test_results["results"][-50:],  # Last 50 results for UI
        "stats": test_results["stats"],
        "distribution": test_results["distribution"],
    }


@app.post("/api/run-test")
async def run_test(config: TestConfig, background_tasks: BackgroundTasks):
    """Start a new latency test."""
    if test_results["status"] == "running":
        return {"error": "Test already running"}
    
    if config.num_requests < 1 or config.num_requests > 1000:
        return {"error": "Number of requests must be between 1 and 1000"}
    
    background_tasks.add_task(execute_test, config)
    return {"message": "Test started"}


def execute_test(config: TestConfig):
    """Execute the latency test."""
    global test_results
    
    test_results = {
        "status": "running",
        "progress": 0,
        "total_requests": config.num_requests,
        "completed_requests": 0,
        "results": [],
        "stats": None,
        "distribution": None,
    }
    
    for i in range(config.num_requests):
        query = TEST_QUERIES[i % len(TEST_QUERIES)]
        result = test_linkup(query, config.depth)
        test_results["results"].append(result)
        test_results["completed_requests"] = i + 1
        test_results["progress"] = int(((i + 1) / config.num_requests) * 100)
        
        # Update stats every 10 requests or at the end
        if (i + 1) % 10 == 0 or i == config.num_requests - 1:
            test_results["stats"] = calculate_stats(test_results["results"])
            test_results["distribution"] = calculate_distribution(test_results["results"])
        
        # Small delay to avoid rate limiting (adjust as needed)
        time.sleep(0.1)
    
    test_results["status"] = "completed"
    test_results["progress"] = 100


@app.post("/api/reset")
async def reset_test():
    """Reset test results."""
    global test_results
    test_results = {
        "status": "idle",
        "progress": 0,
        "total_requests": 0,
        "completed_requests": 0,
        "results": [],
        "stats": None,
        "distribution": None,
    }
    return {"message": "Reset complete"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
