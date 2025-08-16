#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
환 리스크 헷지를 위한 강화학습 에이전트
PPO 알고리즘 기반으로 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class HedgeActorCritic(nn.Module):
    """헷지 전략을 위한 Actor-Critic 신경망"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(HedgeActorCritic, self).__init__()
        
        # 공통 특징 추출 레이어
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor (정책) 네트워크
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic (가치) 네트워크
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파"""
        features = self.feature_extractor(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """액션과 가치 반환"""
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value

class HedgeAgent:
    """환 리스크 헷지 강화학습 에이전트"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'auto'):
        """
        Args:
            state_dim: 상태 차원
            action_dim: 액션 차원
            learning_rate: 학습률
            gamma: 할인 계수
            gae_lambda: GAE 람다
            clip_ratio: PPO 클립 비율
            value_loss_coef: 가치 손실 계수
            entropy_coef: 엔트로피 계수
            max_grad_norm: 최대 그래디언트 노름
            device: 디바이스
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = HedgeActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 학습 히스토리
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """액션 선택"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor)
            
            if training:
                # 학습시에는 확률적 액션 선택
                dist = Categorical(action_probs)
                action = dist.sample()
            else:
                # 평가시에는 최적 액션 선택
                action = torch.argmax(action_probs, dim=-1)
        
        return action.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        """Generalized Advantage Estimation (GAE) 계산"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, states: List[np.ndarray], actions: List[int], 
               rewards: List[float], dones: List[bool], 
               log_probs: List[torch.Tensor]) -> Dict[str, float]:
        """모델 업데이트 (PPO)"""
        # GAE 계산
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            _, values = self.model(states_tensor)
            values = values.squeeze().cpu().numpy()
        
        advantages = self.compute_gae(rewards, values, dones)
        returns = np.array(advantages) + np.array(values)
        
        # 정규화
        advantages = (np.array(advantages) - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # 텐서 변환
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        old_log_probs_tensor = torch.stack(log_probs).detach()
        
        # PPO 업데이트
        for _ in range(10):  # 여러 번 업데이트
            action_probs, values = self.model(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy()
            
            # 비율 계산
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # 클립된 목적 함수
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 가치 손실
            value_loss = F.mse_loss(values.squeeze(), returns_tensor)
            
            # 엔트로피 손실
            entropy_loss = -entropy.mean()
            
            # 전체 손실
            total_loss = (actor_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # 역전파
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 손실 기록
        losses = {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return losses
    
    def train_episode(self, env) -> Tuple[float, int, Dict[str, float]]:
        """에피소드 학습"""
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 에피소드 데이터 수집
        states, actions, rewards, dones, log_probs = [], [], [], [], []
        
        while True:
            # 액션 선택
            action = self.select_action(state, training=True)
            
            # 환경에서 스텝 실행
            next_state, reward, done, truncated, info = env.step(action)
            
            # 데이터 저장
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            # 로그 확률 계산
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([action]).to(self.device)
            with torch.no_grad():
                _, log_prob, _, _ = self.model.get_action_and_value(state_tensor, action_tensor)
            log_probs.append(log_prob.squeeze())
            
            # 상태 업데이트
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done or truncated:
                break
        
        # 모델 업데이트
        losses = self.update(states, actions, rewards, dones, log_probs)
        
        return episode_reward, episode_length, losses
    
    def train(self, env, num_episodes: int = 1000, eval_interval: int = 100) -> Dict[str, List]:
        """학습 실행"""
        print(f"🎯 {num_episodes} 에피소드 학습 시작...")
        
        for episode in tqdm(range(num_episodes), desc="학습 진행률"):
            # 에피소드 학습
            episode_reward, episode_length, losses = self.train_episode(env)
            
            # 히스토리 저장
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['actor_losses'].append(losses['actor_loss'])
            self.training_history['critic_losses'].append(losses['critic_loss'])
            self.training_history['entropy_losses'].append(losses['entropy_loss'])
            self.training_history['total_losses'].append(losses['total_loss'])
            
            # 평가 및 출력
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-eval_interval:])
                avg_length = np.mean(self.training_history['episode_lengths'][-eval_interval:])
                avg_loss = np.mean(self.training_history['total_losses'][-eval_interval:])
                
                print(f"📊 에피소드 {episode + 1}: "
                      f"평균 보상: {avg_reward:.2f}, "
                      f"평균 길이: {avg_length:.1f}, "
                      f"평균 손실: {avg_loss:.4f}")
        
        print("✅ 학습 완료!")
        return self.training_history
    
    def evaluate(self, env, num_episodes: int = 100) -> Dict[str, float]:
        """에이전트 평가"""
        print(f"🔍 {num_episodes} 에피소드 평가 중...")
        
        episode_rewards = []
        episode_lengths = []
        final_capitals = []
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 평가시에는 최적 액션 선택
                action = self.select_action(state, training=False)
                next_state, reward, done, truncated, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            final_capitals.append(info['current_capital'])
        
        # 평가 결과
        evaluation_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_final_capital': np.mean(final_capitals),
            'std_final_capital': np.std(final_capitals),
            'sharpe_ratio': np.mean(episode_rewards) / (np.std(episode_rewards) + 1e-8)
        }
        
        print(f"📈 평가 결과:")
        print(f"   평균 보상: {evaluation_results['mean_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
        print(f"   평균 길이: {evaluation_results['mean_length']:.1f}")
        print(f"   평균 최종 자본: ${evaluation_results['mean_final_capital']:,.2f}")
        print(f"   샤프 비율: {evaluation_results['sharpe_ratio']:.4f}")
        
        return evaluation_results
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        print(f"💾 모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        print(f"📂 모델 로드 완료: {filepath}")
    
    def plot_training_history(self, save_path: str = None) -> None:
        """학습 히스토리 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 에피소드 보상
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('에피소드 보상')
        axes[0, 0].set_xlabel('에피소드')
        axes[0, 0].set_ylabel('보상')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 에피소드 길이
        axes[0, 1].plot(self.training_history['episode_lengths'])
        axes[0, 1].set_title('에피소드 길이')
        axes[0, 1].set_xlabel('에피소드')
        axes[0, 1].set_ylabel('길이')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 손실 함수들
        axes[1, 0].plot(self.training_history['actor_losses'], label='Actor Loss', alpha=0.7)
        axes[1, 0].plot(self.training_history['critic_losses'], label='Critic Loss', alpha=0.7)
        axes[1, 0].plot(self.training_history['entropy_losses'], label='Entropy Loss', alpha=0.7)
        axes[1, 0].set_title('손실 함수')
        axes[1, 0].set_xlabel('에피소드')
        axes[1, 0].set_ylabel('손실')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 전체 손실
        axes[1, 1].plot(self.training_history['total_losses'])
        axes[1, 1].set_title('전체 손실')
        axes[1, 1].set_xlabel('에피소드')
        axes[1, 1].set_ylabel('손실')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 학습 히스토리 저장: {save_path}")
        
        plt.show()
