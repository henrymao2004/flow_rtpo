import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ConvergenceMonitor:
    """
    收敛监控器，用于判断训练是否收敛
    
    实现以下指标：
    1. 滑动平均奖励 (EMA)
    2. 滑动标准差
    3. CVaR-10 (Conditional Value at Risk)
    4. KL散度监控
    5. 收敛判断逻辑
    """
    
    def __init__(
        self,
        ema_decay: float = 0.97,
        convergence_threshold: float = 1e-3,
        patience: int = 3,
        cvar_percentile: float = 0.1,
        kl_stable_range: Tuple[float, float] = (0.01, 0.05),
        std_convergence_ratio: float = 0.3,
        min_epochs: int = 10
    ):
        """
        Args:
            ema_decay: EMA衰减因子，τ ~ 0.97 ⇒ 30–40 步半衰期
            convergence_threshold: 收敛阈值 ε
            patience: 连续多少个epoch没有提升就early-stop
            cvar_percentile: CVaR计算的百分位数
            kl_stable_range: KL散度的稳定范围 [min, max]
            std_convergence_ratio: 标准差收敛到训练早期的比例
            min_epochs: 最少训练epoch数
        """
        self.ema_decay = ema_decay
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.cvar_percentile = cvar_percentile
        self.kl_stable_range = kl_stable_range
        self.std_convergence_ratio = std_convergence_ratio
        self.min_epochs = min_epochs
        
        # 状态变量
        self.state = {
            "ema": 0.0,
            "ema_std": 0.0,
            "step": 0,
            "best_ema": -1e9,
            "since_improve": 0,
            "early_std_values": [],  # 存储训练早期的标准差
            "kl_history": [],
            "cvar_history": [],
            "reward_history": []
        }
        
        # 收敛状态
        self.convergence_status = {
            "is_converged": False,
            "convergence_reason": 0,  # 0: not converged, 1: converged, 2: false convergence
            "epoch": 0
        }
    
    def update(self, rewards: List[float], kl_div: Optional[float] = None, epoch: int = 0) -> Dict:
        """
        更新收敛监控状态
        
        Args:
            rewards: 当前epoch的奖励列表
            kl_div: KL散度值
            epoch: 当前epoch数
            
        Returns:
            包含所有监控指标的字典
        """
        if not rewards:
            return {}
        
        rewards = np.array(rewards)
        current_reward = np.mean(rewards)
        
        # 更新EMA
        self.state["step"] += 1
        prev_ema = self.state["ema"]
        self.state["ema"] = (
            self.ema_decay * prev_ema + 
            (1 - self.ema_decay) * current_reward
        )
        
        # 更新EMA标准差
        self.state["ema_std"] = (
            self.ema_decay * self.state["ema_std"] + 
            (1 - self.ema_decay) * (current_reward - self.state["ema"])**2
        )
        
        # 计算CVaR
        sorted_rewards = np.sort(rewards)
        cvar_idx = max(1, int(self.cvar_percentile * len(sorted_rewards)))
        cvar = np.mean(sorted_rewards[:cvar_idx])
        
        # 存储历史数据
        self.state["reward_history"].append(current_reward)
        self.state["cvar_history"].append(cvar)
        
        if kl_div is not None:
            self.state["kl_history"].append(kl_div)
        
        # 存储训练早期的标准差（前10个epoch）
        if epoch < 10:
            self.state["early_std_values"].append(np.std(rewards))
        
        # 计算指标
        metrics = {
            "ema_reward": self.state["ema"],
            "ema_reward_std": np.sqrt(self.state["ema_std"]),
            "cvar_reward": cvar,
            "reward_mean": current_reward,
            "reward_std": np.std(rewards),
            "ema_change": self.state["ema"] - prev_ema,
            "epoch": epoch
        }
        
        if kl_div is not None:
            metrics["kl_divergence"] = kl_div
        
        # 检查收敛状态
        self._check_convergence(metrics, epoch)
        metrics.update(self.convergence_status)
        
        return metrics
    
    def _check_convergence(self, metrics: Dict, epoch: int):
        """检查是否收敛"""
        if epoch < self.min_epochs:
            return
        
        ema_change = abs(metrics["ema_change"])
        ema_std = metrics["ema_reward_std"]
        current_ema = metrics["ema_reward"]
        
        # 检查EMA是否稳定
        ema_stable = ema_change < self.convergence_threshold
        
        # 检查标准差是否收敛
        std_converged = False
        if len(self.state["early_std_values"]) >= 5:
            early_std_mean = np.mean(self.state["early_std_values"][:5])
            std_converged = ema_std < early_std_mean * self.std_convergence_ratio
        
        # 检查KL散度是否稳定
        kl_stable = True
        if len(self.state["kl_history"]) >= 3:
            recent_kl = np.mean(self.state["kl_history"][-3:])
            kl_stable = (
                self.kl_stable_range[0] <= recent_kl <= self.kl_stable_range[1]
            )
        
        # 检查CVaR是否稳定
        cvar_stable = True
        if len(self.state["cvar_history"]) >= 3:
            recent_cvar = np.mean(self.state["cvar_history"][-3:])
            prev_cvar = np.mean(self.state["cvar_history"][-6:-3]) if len(self.state["cvar_history"]) >= 6 else recent_cvar
            cvar_change = abs(recent_cvar - prev_cvar) / (abs(prev_cvar) + 1e-8)
            cvar_stable = cvar_change < 0.05  # 5%变化阈值
        
        # 检查EMA是否提升
        if current_ema > self.state["best_ema"] + self.convergence_threshold:
            self.state["best_ema"] = current_ema
            self.state["since_improve"] = 0
        else:
            self.state["since_improve"] += 1
        
        # 综合判断收敛
        if (ema_stable and std_converged and kl_stable and cvar_stable and 
            self.state["since_improve"] >= self.patience):
            self.convergence_status.update({
                "is_converged": True,
                "convergence_reason": 1,  # 1: converged
                "epoch": epoch
            })
            logger.info(f"Training converged at epoch {epoch}")
        
        # 检查假收敛
        self._check_false_convergence(metrics, epoch)
    
    def _check_false_convergence(self, metrics: Dict, epoch: int):
        """检查假收敛情况"""
        warnings = []
        
        # 检查KL散度是否过低（策略塌缩）
        if len(self.state["kl_history"]) >= 3:
            recent_kl = np.mean(self.state["kl_history"][-3:])
            if recent_kl < 0.001:
                warnings.append("KL divergence too low - possible policy collapse")
        
        # 检查标准差下降过快但奖励不上升
        if len(self.state["early_std_values"]) >= 5:
            early_std_mean = np.mean(self.state["early_std_values"][:5])
            current_std = metrics["ema_reward_std"]
            if (current_std < early_std_mean * 0.1 and 
                metrics["ema_change"] < self.convergence_threshold):
                warnings.append("Standard deviation dropped too fast without reward improvement")
        
        # 检查CVaR是否还在高位
        if len(self.state["cvar_history"]) >= 3:
            recent_cvar = np.mean(self.state["cvar_history"][-3:])
            if recent_cvar < -0.5:  # 假设奖励范围在[-1, 1]
                warnings.append("CVaR still high - model may have learned high-variance strategy")
        
        if warnings:
            logger.warning(f"Potential false convergence detected at epoch {epoch}: {', '.join(warnings)}")
    
    def should_early_stop(self) -> bool:
        """判断是否应该提前停止训练"""
        return self.convergence_status["is_converged"]
    
    def get_convergence_summary(self) -> Dict:
        """获取收敛摘要"""
        return {
            "convergence_status": self.convergence_status,
            "best_ema": self.state["best_ema"],
            "since_improve": self.state["since_improve"],
            "total_steps": self.state["step"]
        }
    
    def get_convergence_reason_text(self) -> str:
        """获取收敛原因的文本描述"""
        reason_code = self.convergence_status.get("convergence_reason", 0)
        reason_map = {
            0: "Not converged",
            1: "All metrics converged",
            2: "False convergence detected"
        }
        return reason_map.get(reason_code, "Unknown")
    
    def reset(self):
        """重置监控器状态"""
        self.state = {
            "ema": 0.0,
            "ema_std": 0.0,
            "step": 0,
            "best_ema": -1e9,
            "since_improve": 0,
            "early_std_values": [],
            "kl_history": [],
            "cvar_history": [],
            "reward_history": []
        }
        self.convergence_status = {
            "is_converged": False,
            "convergence_reason": 0,  # 0: not converged, 1: converged, 2: false convergence
            "epoch": 0
        }


def calculate_convergence_metrics(
    rewards: List[float], 
    kl_div: Optional[float] = None,
    epoch: int = 0,
    monitor: Optional[ConvergenceMonitor] = None
) -> Dict:
    """
    计算收敛指标的便捷函数
    
    Args:
        rewards: 奖励列表
        kl_div: KL散度
        epoch: 当前epoch
        monitor: 收敛监控器实例
        
    Returns:
        包含所有指标的字典
    """
    if monitor is None:
        monitor = ConvergenceMonitor()
    
    return monitor.update(rewards, kl_div, epoch) 