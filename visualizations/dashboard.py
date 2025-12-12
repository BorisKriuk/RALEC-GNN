"""Dashboard visualization for RALEC-GNN system."""

import os
from typing import Dict, Any, List
import json
from datetime import datetime


class SystemDashboard:
    """Create dashboard visualizations and reports."""
    
    def __init__(self):
        self.template = self._load_template()
        
    def create_evaluation_report(
        self,
        metrics: Dict[str, float],
        outputs: Dict[str, Any],
        save_path: str = "output/evaluation_report.html"
    ):
        """Create HTML evaluation report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RALEC-GNN Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .metric-box {{ 
            display: inline-block; 
            margin: 10px;
            padding: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{ 
            font-size: 36px; 
            font-weight: bold; 
            color: #2ecc71;
        }}
        .metric-label {{ 
            font-size: 14px; 
            color: #666;
            margin-top: 5px;
        }}
        .risk-high {{ color: #e74c3c !important; }}
        .risk-medium {{ color: #f39c12 !important; }}
        table {{ 
            border-collapse: collapse; 
            width: 100%;
            margin-top: 20px;
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px;
            text-align: left;
        }}
        th {{ 
            background-color: #3498db;
            color: white;
        }}
        .alert-box {{
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
        }}
        .alert-critical {{
            background: #ffebee;
            border-left: 5px solid #f44336;
        }}
        .alert-high {{
            background: #fff8e1;
            border-left: 5px solid #ff9800;
        }}
        .alert-warning {{
            background: #e3f2fd;
            border-left: 5px solid #2196f3;
        }}
    </style>
</head>
<body>
    <h1>RALEC-GNN Evaluation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Performance Metrics</h2>
    <div>
        <div class="metric-box">
            <div class="metric-value">{metrics.get('accuracy', 0):.1%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{metrics.get('crisis_recall', 0):.1%}</div>
            <div class="metric-label">Crisis Recall</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{metrics.get('precision', 0):.1%}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{metrics.get('lead_time', 0):.1f}d</div>
            <div class="metric-label">Lead Time</div>
        </div>
    </div>
"""
        
        # Add risk analysis if available
        if 'risk_analysis' in outputs:
            risk = outputs['risk_analysis']
            risk_class = 'risk-high' if risk['overall_risk'] > 0.7 else 'risk-medium'
            
            html_content += f"""
    <h2>Current Risk Analysis</h2>
    <div class="metric-box">
        <div class="metric-value {risk_class}">{risk['overall_risk']:.1%}</div>
        <div class="metric-label">Overall Systemic Risk</div>
    </div>
    
    <table>
        <tr>
            <th>Risk Component</th>
            <th>Value</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>Network Fragility</td>
            <td>{risk.get('network_fragility', 0):.2%}</td>
            <td>{'Critical' if risk.get('network_fragility', 0) > 0.8 else 'Elevated'}</td>
        </tr>
        <tr>
            <td>Cascade Probability</td>
            <td>{risk.get('cascade_probability', 0):.2%}</td>
            <td>{'High' if risk.get('cascade_probability', 0) > 0.6 else 'Moderate'}</td>
        </tr>
        <tr>
            <td>Herding Index</td>
            <td>{risk.get('herding_index', 0):.2%}</td>
            <td>{'Dangerous' if risk.get('herding_index', 0) > 0.7 else 'Monitored'}</td>
        </tr>
    </table>
"""
        
        # Add alerts if any
        if 'alerts' in outputs and outputs['alerts']:
            html_content += "<h2>Active Alerts</h2>"
            for alert in outputs['alerts']:
                alert_class = f"alert-{alert['level'].lower()}"
                html_content += f"""
    <div class="alert-box {alert_class}">
        <strong>[{alert['level']}] {alert['type']}</strong><br>
        {alert['message']}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(html_content)
            
        print(f"Report saved to: {save_path}")
        
    def update(self, outputs: Dict[str, Any], history: List[Any]):
        """Update live dashboard (placeholder for real-time viz)."""
        # In a real implementation, this would update a live dashboard
        # For now, just print status
        if 'risk_analysis' in outputs:
            risk = outputs['risk_analysis']['overall_risk']
            print(f"\r[LIVE] Risk Level: {risk:.1%} ", end="", flush=True)
            
    def _load_template(self) -> str:
        """Load dashboard HTML template."""
        # Simplified - in practice would load from template file
        return ""