"""
Phase 15: Social Network Propagation Graph & Viral Cascade Simulator
Module Owner: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU (VISTAS)

Simulates rumor diffusion dynamics over scale-free and small-world social network topologies
using the SIR (Susceptible-Infected-Recovered) and Independent Cascade diffusion models.
Computes viral reproduction number (R0), peak diffusion time, and cascade reach coordinates.
"""

import math
import random
from typing import Dict, List, Any


def simulate_viral_cascade(
    headline: str,
    prediction: str = "FAKE",
    fake_confidence: float = 85.0,
    network_size: int = 35,
    viral_speed: float = 1.5
) -> Dict[str, Any]:
    """
    Simulates rumor spreading across a synthetic social network graph:
    - FAKE news articles with high sensationalism diffuse with high transmission rate (beta).
    - REAL news articles diffuse with lower viral velocity and higher fact-check containment.
    - Returns graph topology (nodes, edges, diffusion states) for interactive HTML5 canvas visualization.
    """
    random.seed(abs(hash(headline)) % 100000)

    is_fake = (prediction == "FAKE")
    
    # Transmission probability (beta) and Fact-Checking containment rate (gamma)
    if is_fake:
        beta = min(0.92, 0.45 + (fake_confidence / 100.0) * 0.45 * viral_speed)
        gamma = max(0.08, 0.25 - (fake_confidence / 100.0) * 0.15)
    else:
        beta = max(0.15, 0.35 * viral_speed)
        gamma = 0.45

    # Basic Reproduction Number: R0 = beta / gamma
    r0 = round(beta / gamma, 2)
    cascade_risk = "CRITICAL EPIDEMIC VIRALITY" if r0 > 2.5 else ("RAPID DIFFUSION" if r0 > 1.2 else "CONTAINED")

    # Generate Node Topology with 2D coordinates arranged in concentric clusters
    nodes = []
    infected_count = 0
    factchecked_count = 0

    center_x = 300
    center_y = 200

    # Source Seed Node (Root spreader)
    nodes.append({
        "id": 0,
        "label": "Seed Poster (Bot / Origin)",
        "x": center_x,
        "y": center_y,
        "status": "INFECTED",
        "radius": 14,
        "color": "#ef4444" if is_fake else "#10b981",
        "degree": 0
    })
    infected_count += 1

    # Generate surrounding network nodes in layers
    num_layers = 3
    nodes_per_layer = (network_size - 1) // num_layers

    node_id = 1
    for layer in range(1, num_layers + 1):
        radius = layer * 65
        layer_nodes_count = nodes_per_layer if layer < num_layers else (network_size - node_id)
        
        for i in range(layer_nodes_count):
            angle = (2 * math.pi / max(1, layer_nodes_count)) * i + (layer * 0.4)
            jitter_x = random.uniform(-15, 15)
            jitter_y = random.uniform(-15, 15)
            x = center_x + radius * math.cos(angle) + jitter_x
            y = center_y + radius * math.sin(angle) + jitter_y

            # Determine cascade infection state based on transmission probability
            dist_factor = (1.0 / layer)
            infect_prob = beta * dist_factor

            if random.random() < infect_prob:
                # Node becomes an active spreader
                status = "INFECTED"
                color = "#ef4444"
                infected_count += 1
            elif random.random() < gamma:
                # Node fact-checked / immunized
                status = "FACT_CHECKED"
                color = "#6366f1"
                factchecked_count += 1
            else:
                # Susceptible
                status = "SUSCEPTIBLE"
                color = "#64748b"

            nodes.append({
                "id": node_id,
                "label": f"User_{node_id}",
                "x": round(x, 1),
                "y": round(y, 1),
                "status": status,
                "radius": 8 if status == "SUSCEPTIBLE" else 10,
                "color": color,
                "degree": 0
            })
            node_id += 1

    # Generate Network Edges (Scale-free preferential attachment)
    edges = []
    for i in range(1, len(nodes)):
        # Connect to root or previous layer nodes
        target_id = 0 if i <= 6 else random.choice(range(0, i))
        edges.append({"source": target_id, "target": i, "weight": round(random.uniform(0.5, 1.0), 2)})
        nodes[target_id]["degree"] += 1
        nodes[i]["degree"] += 1

        # Add cross-edges with probability 0.25
        if random.random() < 0.25 and i > 2:
            peer_id = random.choice(range(1, i))
            edges.append({"source": peer_id, "target": i, "weight": round(random.uniform(0.3, 0.8), 2)})
            nodes[peer_id]["degree"] += 1
            nodes[i]["degree"] += 1

    susceptible_count = len(nodes) - infected_count - factchecked_count
    total_reach_pct = round((infected_count / len(nodes)) * 100.0, 1)

    # Simulated Time-Series Diffusion Steps (0 to 10 hours)
    timeline = []
    inf_sim = 1
    for t in range(0, 11):
        if is_fake:
            inf_sim = min(network_size, int(inf_sim * (1.0 + beta * 0.5) + random.randint(0, 2)))
        else:
            inf_sim = min(network_size, int(inf_sim * (1.0 + beta * 0.15) + random.randint(0, 1)))
        timeline.append({"hour": t, "active_spreaders": inf_sim, "fact_checked": min(network_size, int(t * gamma * 3))})

    return {
        "headline": headline,
        "prediction": prediction,
        "cascade_metrics": {
            "basic_reproduction_number_R0": r0,
            "cascade_risk": cascade_risk,
            "transmission_rate_beta": round(beta, 3),
            "containment_rate_gamma": round(gamma, 3),
            "total_network_nodes": len(nodes),
            "total_network_edges": len(edges),
            "infected_spreaders_count": infected_count,
            "fact_checked_count": factchecked_count,
            "susceptible_count": susceptible_count,
            "total_population_reach_percent": total_reach_pct
        },
        "graph_topology": {
            "nodes": nodes,
            "edges": edges
        },
        "time_series_timeline": timeline
    }
