"""LangGraph Workflow Visualization with All Sub-Agents as Individual Nodes"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Font settings for English only
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_detailed_langgraph_visualization():
    """Create detailed LangGraph workflow visualization with all sub-agents as nodes"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define all nodes including sub-agents
    nodes = {
        # Control nodes
        'START': {'pos': (7, 11.5), 'color': '#90EE90', 'shape': 'ellipse', 'label': 'START'},
        'reask': {'pos': (7, 10), 'color': '#87CEEB', 'shape': 'rect', 'label': 'ReAsk Agent\nQuery Clarity Check'},
        'routing': {'pos': (7, 8.5), 'color': '#DDA0DD', 'shape': 'rect', 'label': 'Routing Node\nLLM-based Agent Selection'},
        'handsoff': {'pos': (2, 6), 'color': '#FFA07A', 'shape': 'rect', 'label': 'HandsOff Agent\nUser Delivery Decision'},
        'final_answer': {'pos': (7, 2.5), 'color': '#98FB98', 'shape': 'rect', 'label': 'Final Answer Node\nAnswer Generation'},
        'END1': {'pos': (1, 10), 'color': '#FFB6C1', 'shape': 'ellipse', 'label': 'END\n(Clarification)'},
        'END2': {'pos': (7, 1), 'color': '#FFB6C1', 'shape': 'ellipse', 'label': 'END\n(Answer)'},
        
        # Sub-agents (executable agents)
        'market_info': {'pos': (4, 6.5), 'color': '#FFE4B5', 'shape': 'rect', 'label': 'Market Information\nAgent'},
        'expert_info': {'pos': (5.5, 6.5), 'color': '#E0E0E0', 'shape': 'rect', 'label': 'Expert Information\nAgent (RAG)'},
        'user_info': {'pos': (7, 6.5), 'color': '#B0E0E6', 'shape': 'rect', 'label': 'User Information\nAgent'},
        'strategy_exec': {'pos': (8.5, 6.5), 'color': '#D8BFD8', 'shape': 'rect', 'label': 'Strategy Execute\nAgent'},
        'react': {'pos': (10, 6.5), 'color': '#F5DEB3', 'shape': 'rect', 'label': 'ReAct Agent\nReasoning & Acting'},
    }
    
    # Draw nodes
    for node_name, node_info in nodes.items():
        x, y = node_info['pos']
        color = node_info['color']
        shape = node_info['shape']
        label = node_info.get('label', node_name)
        
        if shape == 'ellipse':
            # Ellipse (START, END)
            ellipse = mpatches.Ellipse((x, y), 1.0, 0.5, 
                                      facecolor=color, 
                                      edgecolor='black', 
                                      linewidth=2,
                                      zorder=3)
            ax.add_patch(ellipse)
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=9, fontweight='bold', zorder=4)
        else:
            # Rectangle (regular nodes)
            width = 1.8 if node_name in ['routing', 'final_answer'] else 1.5
            height = 0.6 if node_name in ['routing', 'final_answer'] else 0.5
            box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                boxstyle="round,pad=0.1",
                                facecolor=color,
                                edgecolor='black',
                                linewidth=2,
                                zorder=3)
            ax.add_patch(box)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=8, fontweight='bold', zorder=4)
    
    # Define edges - LangGraph structure
    edges = [
        # Main flow
        ('START', 'reask', 'black', 'solid', None),
        ('reask', 'END1', '#FF6B6B', 'dashed', 'needs_clarification'),
        ('reask', 'routing', '#4ECDC4', 'solid', 'continue'),
        
        # Routing to agents (conditional - only selected agents execute)
        ('routing', 'market_info', '#4169E1', 'solid', 'if selected'),
        ('routing', 'expert_info', '#4169E1', 'solid', 'if selected'),
        ('routing', 'user_info', '#4169E1', 'solid', 'if selected'),
        ('routing', 'strategy_exec', '#4169E1', 'solid', 'if selected'),
        ('routing', 'react', '#4169E1', 'solid', 'if selected'),
        
        # Sequential execution flow (agents execute in order)
        ('market_info', 'expert_info', '#808080', 'dotted', 'next'),
        ('expert_info', 'user_info', '#808080', 'dotted', 'next'),
        ('user_info', 'strategy_exec', '#808080', 'dotted', 'next'),
        ('strategy_exec', 'react', '#808080', 'dotted', 'next'),
        
        # After each agent: check for direct answer or hands-off
        ('market_info', 'final_answer', '#32CD32', 'dashed', 'has answer'),
        ('expert_info', 'final_answer', '#32CD32', 'dashed', 'has answer'),
        ('user_info', 'final_answer', '#32CD32', 'dashed', 'has answer'),
        ('strategy_exec', 'final_answer', '#32CD32', 'dashed', 'has answer'),
        ('react', 'final_answer', '#32CD32', 'dashed', 'has answer'),
        
        # After each agent: hands-off check (if no direct answer)
        ('market_info', 'handsoff', '#9370DB', 'dashed', 'check'),
        ('expert_info', 'handsoff', '#9370DB', 'dashed', 'check'),
        ('user_info', 'handsoff', '#9370DB', 'dashed', 'check'),
        ('strategy_exec', 'handsoff', '#9370DB', 'dashed', 'check'),
        ('react', 'handsoff', '#9370DB', 'dashed', 'check'),
        
        # HandsOff decisions
        ('handsoff', 'final_answer', '#4ECDC4', 'solid', 'forward'),
        ('handsoff', 'routing', '#FFA500', 'dotted', 'continue (loop)'),
        
        # Final answer to END
        ('final_answer', 'END2', 'black', 'solid', None),
    ]
    
    # Draw edges
    for start, end, color, style, label in edges:
        start_pos = nodes[start]['pos']
        end_pos = nodes[end]['pos']
        
        # Special handling for loop edge
        if start == 'handsoff' and end == 'routing':
            # Curved arrow for loop
            arrow = FancyArrowPatch((start_pos[0]-0.5, start_pos[1]), 
                                   (end_pos[0]-0.5, end_pos[1]),
                                   connectionstyle="arc3,rad=0.4",
                                   arrowstyle='->',
                                   color=color,
                                   linestyle=style,
                                   linewidth=2,
                                   zorder=2)
            ax.add_patch(arrow)
            # Label
            ax.text(1, 7.5, label, ha='left', va='center',
                   fontsize=7, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            # Straight arrow
            arrow = FancyArrowPatch(start_pos, end_pos,
                                   arrowstyle='->',
                                   color=color,
                                   linestyle=style,
                                   linewidth=1.5,
                                   zorder=2,
                                   alpha=0.7)
            ax.add_patch(arrow)
            
            # Label for important edges
            if label and label not in ['next', 'check', 'if selected']:
                mid_x = (start_pos[0] + end_pos[0]) / 2
                mid_y = (start_pos[1] + end_pos[1]) / 2
                
                # Offset for better visibility
                if start == 'reask' and end == 'END1':
                    mid_x -= 0.5
                    mid_y += 0.2
                elif start == 'reask' and end == 'routing':
                    mid_x += 0.5
                elif start == 'handsoff' and end == 'final_answer':
                    mid_x += 0.3
                    mid_y -= 0.2
                
                ax.text(mid_x, mid_y, label, ha='center', va='center',
                       fontsize=7, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    # Title
    ax.text(7, 11.8, 'LangGraph Workflow - FX Hedge Agent Supervisor', 
           ha='center', va='bottom', fontsize=18, fontweight='bold')
    ax.text(7, 11.5, 'All Sub-Agents as Individual Nodes', 
           ha='center', va='top', fontsize=12, style='italic', color='#666666')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='START/END'),
        mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='ReAsk Agent'),
        mpatches.Patch(facecolor='#DDA0DD', edgecolor='black', label='Routing Node'),
        mpatches.Patch(facecolor='#FFE4B5', edgecolor='black', label='Market Info Agent'),
        mpatches.Patch(facecolor='#E0E0E0', edgecolor='black', label='Expert Info Agent'),
        mpatches.Patch(facecolor='#B0E0E6', edgecolor='black', label='User Info Agent'),
        mpatches.Patch(facecolor='#D8BFD8', edgecolor='black', label='Strategy Execute Agent'),
        mpatches.Patch(facecolor='#F5DEB3', edgecolor='black', label='ReAct Agent'),
        mpatches.Patch(facecolor='#FFA07A', edgecolor='black', label='HandsOff Agent'),
        mpatches.Patch(facecolor='#98FB98', edgecolor='black', label='Final Answer Node'),
        plt.Line2D([0], [0], color='black', linestyle='solid', label='Main Flow'),
        plt.Line2D([0], [0], color='#4169E1', linestyle='solid', label='Routing to Agent'),
        plt.Line2D([0], [0], color='#808080', linestyle='dotted', label='Sequential Execution'),
        plt.Line2D([0], [0], color='#9370DB', linestyle='dashed', label='HandsOff Check'),
        plt.Line2D([0], [0], color='#32CD32', linestyle='dashed', label='Direct to Answer'),
        plt.Line2D([0], [0], color='#FFA500', linestyle='dotted', label='Loop Back'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
    
    # Add annotation box explaining the flow
    annotation_text = (
        "LangGraph Workflow:\n"
        "1. START → ReAsk Agent: Check query clarity\n"
        "2. ReAsk → Routing Node: LLM selects agents\n"
        "3. Routing → Selected Agents: Execute sequentially\n"
        "4. Each Agent → Final Answer: If has direct answer\n"
        "5. Each Agent → HandsOff: If needs decision\n"
        "6. HandsOff → Final Answer: Forward to user\n"
        "7. HandsOff → Routing: Loop if more info needed\n"
        "8. Final Answer → END: Return response"
    )
    ax.text(12, 8, annotation_text, ha='left', va='top', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print("📊 Creating detailed LangGraph workflow visualization...")
    fig = create_detailed_langgraph_visualization()
    
    # Save image
    output_path = 'docs/langgraph_workflow_detailed.png'
    import os
    os.makedirs('docs', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Image saved: {output_path}")
    
    # Save as PDF
    pdf_path = 'docs/langgraph_workflow_detailed.pdf'
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ PDF saved: {pdf_path}")
    
    plt.show()
    print("✅ Visualization complete!")

