"""LangGraph 워크플로우 시각화 스크립트"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 한글 폰트 설정 (macOS)
import platform
if platform.system() == 'Darwin':  # macOS
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except:
            plt.rcParams['font.family'] = 'Arial Unicode MS'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_langgraph_visualization():
    """LangGraph 워크플로우 시각화"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 노드 정의
    nodes = {
        'START': {'pos': (5, 9.5), 'color': '#90EE90', 'shape': 'ellipse'},
        'reask': {'pos': (5, 8), 'color': '#87CEEB', 'shape': 'rect', 'label': 'ReAsk Node\n질의 명확성 확인'},
        'routing': {'pos': (5, 6.5), 'color': '#DDA0DD', 'shape': 'rect', 'label': 'Routing Node\nLLM 기반 에이전트 선택'},
        'agent_exec': {'pos': (5, 5), 'color': '#F0E68C', 'shape': 'rect', 'label': 'Agent Execution Node\n에이전트 순차 실행'},
        'handsoff': {'pos': (2.5, 3.5), 'color': '#FFA07A', 'shape': 'rect', 'label': 'HandsOff Node\n사용자 전달 여부 결정'},
        'final_answer': {'pos': (7.5, 3.5), 'color': '#98FB98', 'shape': 'rect', 'label': 'Final Answer Node\n최종 답변 생성'},
        'END1': {'pos': (2, 8), 'color': '#FFB6C1', 'shape': 'ellipse', 'label': 'END\n(재질문)'},
        'END2': {'pos': (7.5, 2), 'color': '#FFB6C1', 'shape': 'ellipse', 'label': 'END\n(답변)'}
    }
    
    # 노드 그리기
    for node_name, node_info in nodes.items():
        x, y = node_info['pos']
        color = node_info['color']
        shape = node_info['shape']
        label = node_info.get('label', node_name)
        
        if shape == 'ellipse':
            # 타원형 (START, END)
            ellipse = mpatches.Ellipse((x, y), 1.2, 0.6, 
                                      facecolor=color, 
                                      edgecolor='black', 
                                      linewidth=2,
                                      zorder=3)
            ax.add_patch(ellipse)
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=10, fontweight='bold', zorder=4)
        else:
            # 사각형 (일반 노드)
            if node_name == 'agent_exec':
                # Agent Execution Node는 더 크게
                box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color,
                                    edgecolor='black',
                                    linewidth=2,
                                    zorder=3)
            else:
                box = FancyBboxPatch((x-1, y-0.3), 2, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color,
                                    edgecolor='black',
                                    linewidth=2,
                                    zorder=3)
            ax.add_patch(box)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=9, fontweight='bold', zorder=4)
    
    # 엣지 그리기
    edges = [
        # START -> reask
        ('START', 'reask', 'black', 'solid', None),
        # reask -> END1 (clarify)
        ('reask', 'END1', '#FF6B6B', 'dashed', 'needs_clarification'),
        # reask -> routing (continue)
        ('reask', 'routing', '#4ECDC4', 'solid', 'continue'),
        # routing -> agent_exec
        ('routing', 'agent_exec', 'black', 'solid', None),
        # agent_exec -> final_answer (final_answer 있음)
        ('agent_exec', 'final_answer', '#4ECDC4', 'solid', 'final_answer 있음'),
        # agent_exec -> handsoff (final_answer 없음)
        ('agent_exec', 'handsoff', '#FF6B6B', 'dashed', 'final_answer 없음'),
        # handsoff -> final_answer (forward)
        ('handsoff', 'final_answer', '#4ECDC4', 'solid', 'forward'),
        # handsoff -> routing (continue, 루프)
        ('handsoff', 'routing', '#FFA500', 'dotted', 'continue (루프)'),
        # final_answer -> END2
        ('final_answer', 'END2', 'black', 'solid', None)
    ]
    
    for start, end, color, style, label in edges:
        start_pos = nodes[start]['pos']
        end_pos = nodes[end]['pos']
        
        # 루프 엣지는 곡선으로
        if start == 'handsoff' and end == 'routing':
            # 곡선 그리기
            mid_x = 1
            mid_y = 5
            arrow = FancyArrowPatch((start_pos[0]-1, start_pos[1]), 
                                   (end_pos[0]-1, end_pos[1]),
                                   connectionstyle="arc3,rad=0.3",
                                   arrowstyle='->',
                                   color=color,
                                   linestyle=style,
                                   linewidth=2,
                                   zorder=2)
            ax.add_patch(arrow)
            # 라벨
            ax.text(mid_x, mid_y, label, ha='left', va='center',
                   fontsize=8, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            # 직선 화살표
            arrow = FancyArrowPatch(start_pos, end_pos,
                                   arrowstyle='->',
                                   color=color,
                                   linestyle=style,
                                   linewidth=2,
                                   zorder=2)
            ax.add_patch(arrow)
            
            # 라벨 (중간 지점)
            if label:
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                # 약간 오프셋
                if start == 'reask' and end == 'END1':
                    mid_x -= 0.5
                    mid_y += 0.2
                elif start == 'reask' and end == 'routing':
                    mid_x += 0.5
                elif start == 'agent_exec' and end == 'final_answer':
                    mid_x += 0.5
                    mid_y -= 0.3
                elif start == 'agent_exec' and end == 'handsoff':
                    mid_x -= 0.5
                    mid_y -= 0.3
                elif start == 'handsoff' and end == 'final_answer':
                    mid_x += 0.3
                    mid_y += 0.2
                
                ax.text(mid_x, mid_y, label, ha='center', va='center',
                       fontsize=8, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 제목
    ax.text(5, 9.8, 'LangGraph Workflow - FX Hedge Agent Supervisor', 
           ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # 범례
    legend_elements = [
        mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='START/END'),
        mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='ReAsk Node'),
        mpatches.Patch(facecolor='#DDA0DD', edgecolor='black', label='Routing Node'),
        mpatches.Patch(facecolor='#F0E68C', edgecolor='black', label='Agent Execution'),
        mpatches.Patch(facecolor='#FFA07A', edgecolor='black', label='HandsOff Node'),
        mpatches.Patch(facecolor='#98FB98', edgecolor='black', label='Final Answer'),
        plt.Line2D([0], [0], color='black', linestyle='solid', label='일반 엣지'),
        plt.Line2D([0], [0], color='#FF6B6B', linestyle='dashed', label='조건부 분기 (종료)'),
        plt.Line2D([0], [0], color='#4ECDC4', linestyle='solid', label='조건부 분기 (진행)'),
        plt.Line2D([0], [0], color='#FFA500', linestyle='dotted', label='루프 엣지')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    # Agent Execution Node에 실행 가능한 에이전트 목록 추가
    agent_list = [
        '• market_information',
        '• expert_information',
        '• user_information',
        '• strategy_execute',
        '• react'
    ]
    agent_text = '\n'.join(agent_list)
    ax.text(5, 4.2, f'실행 가능한 에이전트:\n{agent_text}',
           ha='center', va='top', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("📊 LangGraph 워크플로우 시각화 생성 중...")
    fig = create_langgraph_visualization()
    
    # 이미지 저장
    output_path = 'docs/langgraph_workflow.png'
    import os
    os.makedirs('docs', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 이미지 저장 완료: {output_path}")
    
    # PDF로도 저장
    pdf_path = 'docs/langgraph_workflow.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF 저장 완료: {pdf_path}")
    
    plt.show()
    print("✅ 시각화 완료!")

