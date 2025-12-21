"""
LangGraph工作流模块
构建PCB缺陷检测的多步骤智能体流程：
1. 检测缺陷
2. 检索相似案例
3. 生成维修报告
4. 质量评估
"""
from typing import TypedDict, List, Dict, Any, Annotated
import json
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("警告: LangGraph未安装，将使用简化版本")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.inference.pcb_agent import SimplePCBAgent, PCBDefectInput
from src.inference.vector_store import PCBVectorStore


class PCBInspectionState(TypedDict):
    """PCB检测状态"""
    image_path: str
    inspection_type: str
    defects: List[Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]
    repair_report: str
    quality_score: float
    metadata: Dict[str, Any]


class PCBLangGraphAgent:
    """基于LangGraph的PCB检测智能体"""
    
    def __init__(
        self,
        model_path: str = "./models/qwen3-vl-pcb-awq",
        vector_store: PCBVectorStore = None,
        collection_name: str = "pcb_defects"
    ):
        """
        初始化LangGraph智能体
        
        Args:
            model_path: 模型路径
            vector_store: 向量存储实例（可选）
            collection_name: 向量数据库集合名称
        """
        self.agent = SimplePCBAgent(model_path=model_path)
        self.vector_store = vector_store
        
        if self.vector_store is None:
            from src.inference.vector_store import create_vector_store
            self.vector_store = create_vector_store(collection_name=collection_name)
        
        # 构建工作流图
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
        else:
            self.graph = None
            print("使用简化版本（LangGraph未安装）")
    
    def _build_graph(self):
        """构建LangGraph工作流"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        # 创建状态图
        workflow = StateGraph(PCBInspectionState)
        
        # 添加节点
        workflow.add_node("detect", self._detect_defects)
        workflow.add_node("retrieve", self._retrieve_similar_cases)
        workflow.add_node("generate_report", self._generate_repair_report)
        workflow.add_node("evaluate", self._evaluate_quality)
        workflow.add_node("store_result", self._store_result)
        
        # 定义边
        workflow.set_entry_point("detect")
        workflow.add_edge("detect", "retrieve")
        workflow.add_edge("retrieve", "generate_report")
        workflow.add_edge("generate_report", "evaluate")
        workflow.add_edge("evaluate", "store_result")
        workflow.add_edge("store_result", END)
        
        return workflow.compile()
    
    def _detect_defects(self, state: PCBInspectionState) -> PCBInspectionState:
        """节点1: 检测缺陷"""
        print("🔍 步骤1: 检测缺陷...")
        
        try:
            defects = self.agent.inspect(
                state["image_path"],
                state.get("inspection_type", "full")
            )
            
            # 过滤错误
            defects = [d for d in defects if not d.get("error")]
            
            state["defects"] = defects
            print(f"   发现 {len(defects)} 个缺陷")
        except Exception as e:
            print(f"   检测失败: {e}")
            state["defects"] = []
        
        return state
    
    def _retrieve_similar_cases(
        self,
        state: PCBInspectionState
    ) -> PCBInspectionState:
        """节点2: 检索相似案例"""
        print("📚 步骤2: 检索相似案例...")
        
        defects = state.get("defects", [])
        
        if not defects:
            state["similar_cases"] = []
            print("   无缺陷，跳过检索")
            return state
        
        try:
            similar_cases = self.vector_store.search_similar_defects(
                query_defects=defects,
                top_k=5,
                min_score=0.7
            )
            
            state["similar_cases"] = similar_cases
            print(f"   找到 {len(similar_cases)} 个相似案例")
        except Exception as e:
            print(f"   检索失败: {e}")
            state["similar_cases"] = []
        
        return state
    
    def _generate_repair_report(
        self,
        state: PCBInspectionState
    ) -> PCBInspectionState:
        """节点3: 生成维修报告"""
        print("📝 步骤3: 生成维修报告...")
        
        defects = state.get("defects", [])
        similar_cases = state.get("similar_cases", [])
        
        # 构建报告
        report_parts = []
        report_parts.append(f"# PCB缺陷检测报告")
        report_parts.append(f"**检测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append(f"**图像路径**: {state['image_path']}")
        report_parts.append("")
        
        if not defects:
            report_parts.append("## 检测结果")
            report_parts.append("✅ **正常，未发现缺陷**")
        else:
            report_parts.append(f"## 检测结果（发现 {len(defects)} 个缺陷）")
            report_parts.append("")
            
            for i, defect in enumerate(defects, 1):
                defect_type = defect.get("defect", "unknown")
                bbox = defect.get("bbox", [])
                repair = defect.get("repair", "")
                confidence = defect.get("confidence", 0.0)
                
                report_parts.append(f"### 缺陷 {i}: {defect_type}")
                report_parts.append(f"- **位置**: {bbox}")
                report_parts.append(f"- **置信度**: {confidence:.2%}")
                report_parts.append(f"- **维修建议**: {repair}")
                report_parts.append("")
            
            # 添加相似案例参考
            if similar_cases:
                report_parts.append("## 相似历史案例参考")
                report_parts.append("")
                for j, case in enumerate(similar_cases[:3], 1):  # 只显示前3个
                    similarity = case.get("similarity", 0)
                    case_defects = json.loads(case.get("defects_json", "[]"))
                    report_parts.append(f"### 案例 {j} (相似度: {similarity:.2%})")
                    if case_defects:
                        report_parts.append(f"- **缺陷类型**: {case_defects[0].get('defect', 'unknown')}")
                        report_parts.append(f"- **历史维修方案**: {case_defects[0].get('repair', '')}")
                    report_parts.append("")
        
        state["repair_report"] = "\n".join(report_parts)
        
        return state
    
    def _evaluate_quality(
        self,
        state: PCBInspectionState
    ) -> PCBInspectionState:
        """节点4: 质量评估"""
        print("⭐ 步骤4: 质量评估...")
        
        defects = state.get("defects", [])
        
        if not defects:
            # 无缺陷，质量分数为1.0
            state["quality_score"] = 1.0
            print("   质量分数: 1.0 (无缺陷)")
            return state
        
        # 根据缺陷数量和类型计算质量分数
        # 简化评估逻辑：每个缺陷扣分
        base_score = 1.0
        defect_penalty = {
            "short": 0.3,   # 短路严重
            "open": 0.25,   # 断路较严重
            "missing": 0.2, # 缺件中等
        }
        
        total_penalty = 0.0
        for defect in defects:
            defect_type = defect.get("defect", "unknown")
            penalty = defect_penalty.get(defect_type, 0.15)
            # 考虑置信度
            confidence = defect.get("confidence", 1.0)
            total_penalty += penalty * confidence
        
        quality_score = max(0.0, base_score - total_penalty)
        state["quality_score"] = quality_score
        
        print(f"   质量分数: {quality_score:.2f}")
        
        return state
    
    def _store_result(
        self,
        state: PCBInspectionState
    ) -> PCBInspectionState:
        """节点5: 存储检测结果"""
        print("💾 步骤5: 存储检测结果...")
        
        try:
            doc_id = self.vector_store.add_detection_result(
                image_path=state["image_path"],
                defects=state.get("defects", []),
                metadata={
                    "inspection_type": state.get("inspection_type", "full"),
                    "quality_score": state.get("quality_score", 0.0),
                    "similar_cases_count": len(state.get("similar_cases", []))
                }
            )
            
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["stored_id"] = doc_id
            print(f"   已存储，ID: {doc_id}")
        except Exception as e:
            print(f"   存储失败: {e}")
        
        return state
    
    def inspect(
        self,
        image_path: str,
        inspection_type: str = "full",
        use_graph: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整的检测流程
        
        Args:
            image_path: 图像路径
            inspection_type: 检测类型
            use_graph: 是否使用LangGraph工作流（如果可用）
        
        Returns:
            完整检测结果
        """
        # 初始化状态
        initial_state: PCBInspectionState = {
            "image_path": image_path,
            "inspection_type": inspection_type,
            "defects": [],
            "similar_cases": [],
            "repair_report": "",
            "quality_score": 0.0,
            "metadata": {}
        }
        
        if use_graph and self.graph is not None:
            # 使用LangGraph工作流
            print("使用LangGraph工作流...")
            final_state = self.graph.invoke(initial_state)
        else:
            # 使用简化版本（顺序执行）
            print("使用简化工作流...")
            final_state = initial_state
            
            # 按顺序执行各个步骤
            final_state = self._detect_defects(final_state)
            final_state = self._retrieve_similar_cases(final_state)
            final_state = self._generate_repair_report(final_state)
            final_state = self._evaluate_quality(final_state)
            final_state = self._store_result(final_state)
        
        return final_state
    
    def get_case_history(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """获取历史案例"""
        stats = self.vector_store.get_statistics()
        total = stats.get("total_cases", 0)
        
        # 简单实现：返回最近的k个案例
        # 实际可以使用更复杂的查询
        return []


if __name__ == "__main__":
    # 测试LangGraph工作流
    print("测试LangGraph工作流...")
    
    agent = PCBLangGraphAgent()
    
    # 执行检测
    result = agent.inspect(
        image_path="test_board.jpg",
        inspection_type="full"
    )
    
    print("\n检测结果:")
    print(f"缺陷数量: {len(result['defects'])}")
    print(f"相似案例: {len(result['similar_cases'])}")
    print(f"质量分数: {result['quality_score']:.2f}")
    print("\n维修报告:")
    print(result['repair_report'])

