"""Risk metrics calculation for RALEC-GNN."""

import numpy as np
from typing import Dict, Any, List


class RiskMetricsCalculator:
    """Calculate various risk metrics for systemic risk assessment."""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
    
    def calculate_metrics(
        self,
        risk_indicators: Dict[str, float],
        historical_indicators: List[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            risk_indicators: Current risk indicators
            historical_indicators: Historical risk data for trend analysis
            
        Returns:
            Dictionary of calculated risk metrics
        """
        metrics = {
            'current_level': self._get_risk_level(risk_indicators.get('overall_risk', 0)),
            'risk_score': risk_indicators.get('overall_risk', 0),
            'components': self._analyze_components(risk_indicators),
            'alerts': self._generate_alerts(risk_indicators),
            'trend': self._analyze_trend(risk_indicators, historical_indicators)
        }
        
        return metrics
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical level."""
        if risk_score < self.risk_thresholds['low']:
            return 'LOW'
        elif risk_score < self.risk_thresholds['medium']:
            return 'MEDIUM'
        elif risk_score < self.risk_thresholds['high']:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _analyze_components(self, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Analyze risk components."""
        components = {}
        
        # Network risk
        network_risk = indicators.get('network_fragility', 0)
        components['network'] = {
            'score': network_risk,
            'level': self._get_risk_level(network_risk),
            'description': self._get_network_description(network_risk)
        }
        
        # Behavioral risk
        behavioral_risk = max(
            indicators.get('herding_index', 0),
            indicators.get('synchronization_risk', 0)
        )
        components['behavioral'] = {
            'score': behavioral_risk,
            'level': self._get_risk_level(behavioral_risk),
            'description': self._get_behavioral_description(behavioral_risk)
        }
        
        # Contagion risk
        contagion_risk = max(
            indicators.get('cascade_probability', 0),
            indicators.get('information_contagion', 0)
        )
        components['contagion'] = {
            'score': contagion_risk,
            'level': self._get_risk_level(contagion_risk),
            'description': self._get_contagion_description(contagion_risk)
        }
        
        return components
    
    def _generate_alerts(self, indicators: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on indicators."""
        alerts = []
        
        # Overall risk alert
        overall_risk = indicators.get('overall_risk', 0)
        if overall_risk > self.risk_thresholds['critical']:
            alerts.append({
                'level': 'CRITICAL',
                'type': 'SYSTEMIC_RISK',
                'message': f'Critical systemic risk detected: {overall_risk:.2%}',
                'action': 'Immediate risk reduction required'
            })
        elif overall_risk > self.risk_thresholds['high']:
            alerts.append({
                'level': 'HIGH',
                'type': 'SYSTEMIC_RISK',
                'message': f'High systemic risk: {overall_risk:.2%}',
                'action': 'Implement defensive measures'
            })
        
        # Component-specific alerts
        if indicators.get('network_fragility', 0) > 0.85:
            alerts.append({
                'level': 'HIGH',
                'type': 'NETWORK_FRAGILITY',
                'message': 'Network structure approaching critical fragility',
                'action': 'Reduce interconnectedness'
            })
        
        if indicators.get('cascade_probability', 0) > 0.7:
            alerts.append({
                'level': 'HIGH',
                'type': 'CASCADE_RISK',
                'message': 'High probability of cascading failures',
                'action': 'Implement circuit breakers'
            })
        
        if indicators.get('herding_index', 0) > 0.8:
            alerts.append({
                'level': 'WARNING',
                'type': 'HERDING',
                'message': 'Dangerous herding behavior detected',
                'action': 'Encourage strategy diversification'
            })
        
        return alerts
    
    def _analyze_trend(
        self,
        current: Dict[str, float],
        historical: List[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Analyze risk trend."""
        if not historical or len(historical) < 2:
            return {'direction': 'STABLE', 'change': 0.0}
        
        # Get recent history
        recent = historical[-10:] if len(historical) >= 10 else historical
        recent_risks = [h.get('overall_risk', 0) for h in recent]
        current_risk = current.get('overall_risk', 0)
        
        # Calculate trend
        if len(recent_risks) >= 2:
            avg_recent = np.mean(recent_risks)
            change = current_risk - avg_recent
            
            if change > 0.1:
                direction = 'INCREASING'
            elif change < -0.1:
                direction = 'DECREASING'
            else:
                direction = 'STABLE'
                
            return {
                'direction': direction,
                'change': float(change),
                'rate': float(change / len(recent)) if len(recent) > 0 else 0.0
            }
        
        return {'direction': 'STABLE', 'change': 0.0}
    
    def _get_network_description(self, risk: float) -> str:
        """Get description for network risk level."""
        if risk < 0.3:
            return "Network structure is robust and well-distributed"
        elif risk < 0.6:
            return "Network showing some concentration, monitor closely"
        elif risk < 0.8:
            return "Network fragility elevated, consider protective measures"
        else:
            return "Network critically fragile, high breakdown risk"
    
    def _get_behavioral_description(self, risk: float) -> str:
        """Get description for behavioral risk level."""
        if risk < 0.3:
            return "Market participants showing healthy diversity"
        elif risk < 0.6:
            return "Some herding behavior emerging, stay vigilant"
        elif risk < 0.8:
            return "Significant synchronization detected, risk elevated"
        else:
            return "Dangerous herding levels, market prone to panics"
    
    def _get_contagion_description(self, risk: float) -> str:
        """Get description for contagion risk level."""
        if risk < 0.3:
            return "Contagion risk minimal, firewalls effective"
        elif risk < 0.6:
            return "Moderate contagion channels forming"
        elif risk < 0.8:
            return "High contagion risk, spillovers likely"
        else:
            return "Extreme contagion risk, cascades imminent"