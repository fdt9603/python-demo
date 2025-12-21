"""
PCB缺陷检测系统使用示例
展示如何使用向量数据库和LangGraph工作流
"""
import os
from pcb_agent import SimplePCBAgent
from vector_store import create_vector_store
from pcb_graph import PCBLangGraphAgent


def example_basic_inspection():
    """示例1: 基础检测（不使用向量数据库）"""
    print("=" * 60)
    print("示例1: 基础检测")
    print("=" * 60)
    
    agent = SimplePCBAgent(model_path="./models/qwen3-vl-pcb-awq")
    
    defects = agent.inspect(
        image_path="data/test_images/board_001.jpg",
        inspection_type="full"
    )
    
    print(f"检测到 {len(defects)} 个缺陷:")
    for i, defect in enumerate(defects, 1):
        print(f"  {i}. {defect.get('defect')} - {defect.get('repair')}")


def example_with_vector_store():
    """示例2: 使用向量数据库存储和检索"""
    print("\n" + "=" * 60)
    print("示例2: 使用向量数据库")
    print("=" * 60)
    
    # 创建向量存储
    vector_store = create_vector_store(
        collection_name="pcb_defects",
        persist_directory="./vector_db"
    )
    
    # 创建带向量存储的智能体
    agent = SimplePCBAgent(
        model_path="./models/qwen3-vl-pcb-awq",
        vector_store=vector_store
    )
    
    # 执行检测（结果会自动保存到向量数据库）
    print("执行检测...")
    defects = agent.inspect(
        image_path="data/test_images/board_001.jpg",
        inspection_type="full"
    )
    
    print(f"检测到 {len(defects)} 个缺陷")
    
    # 搜索相似案例
    if defects:
        print("\n搜索相似案例...")
        similar_cases = agent.search_similar_cases(defects, top_k=3)
        
        print(f"找到 {len(similar_cases)} 个相似案例:")
        for i, case in enumerate(similar_cases, 1):
            similarity = case.get('similarity', 0)
            print(f"  {i}. 相似度: {similarity:.2%}, ID: {case.get('id')}")
    
    # 查看统计信息
    stats = vector_store.get_statistics()
    print(f"\n向量数据库统计: {stats}")


def example_langgraph_workflow():
    """示例3: 使用LangGraph完整工作流"""
    print("\n" + "=" * 60)
    print("示例3: LangGraph完整工作流")
    print("=" * 60)
    
    # 创建LangGraph智能体（自动使用向量数据库）
    agent = PCBLangGraphAgent(
        model_path="./models/qwen3-vl-pcb-awq",
        collection_name="pcb_defects"
    )
    
    # 执行完整工作流
    result = agent.inspect(
        image_path="data/test_images/board_001.jpg",
        inspection_type="full",
        use_graph=True  # 使用LangGraph工作流
    )
    
    # 查看结果
    print(f"\n检测结果:")
    print(f"  缺陷数量: {len(result['defects'])}")
    print(f"  相似案例数: {len(result['similar_cases'])}")
    print(f"  质量分数: {result['quality_score']:.2f}")
    
    # 打印维修报告
    print(f"\n维修报告:")
    print(result['repair_report'])


def example_batch_processing():
    """示例4: 批量处理和历史案例检索"""
    print("\n" + "=" * 60)
    print("示例4: 批量处理")
    print("=" * 60)
    
    vector_store = create_vector_store()
    agent = SimplePCBAgent(
        model_path="./models/qwen3-vl-pcb-awq",
        vector_store=vector_store
    )
    
    # 批量处理图像
    image_dir = "data/test_images"
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ][:5]  # 只处理前5张
    
    print(f"批量处理 {len(image_files)} 张图像...")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\n处理: {image_file}")
        
        defects = agent.inspect(image_path, inspection_type="full")
        print(f"  检测到 {len(defects)} 个缺陷")
        
        if defects:
            similar = agent.search_similar_cases(defects, top_k=1)
            if similar:
                print(f"  最相似案例: {similar[0].get('similarity', 0):.2%}")
    
    # 导出所有数据
    print("\n导出向量数据库...")
    vector_store.export_to_json("exported_cases.json")
    
    stats = vector_store.get_statistics()
    print(f"总共存储了 {stats['total_cases']} 个案例")


def example_advanced_retrieval():
    """示例5: 高级检索功能"""
    print("\n" + "=" * 60)
    print("示例5: 高级检索")
    print("=" * 60)
    
    vector_store = create_vector_store()
    
    # 构建查询缺陷
    query_defects = [
        {
            "defect": "short",
            "bbox": [100, 200, 50, 20],
            "repair": "清理焊锡桥接"
        }
    ]
    
    # 搜索相似案例
    similar_cases = vector_store.search_similar_defects(
        query_defects=query_defects,
        top_k=5,
        min_score=0.7
    )
    
    print(f"查询缺陷: {query_defects[0]['defect']}")
    print(f"找到 {len(similar_cases)} 个相似案例 (相似度 >= 0.7):")
    
    for i, case in enumerate(similar_cases, 1):
        similarity = case.get('similarity', 0)
        case_id = case.get('id')
        defects_json = case.get('defects_json', '[]')
        
        print(f"\n  案例 {i}:")
        print(f"    ID: {case_id}")
        print(f"    相似度: {similarity:.2%}")
        print(f"    图像: {case.get('image_path', 'N/A')}")
        print(f"    检测时间: {case.get('timestamp', 'N/A')}")


if __name__ == "__main__":
    print("PCB缺陷检测系统使用示例")
    print("=" * 60)
    
    # 注意：运行前确保模型已训练并可用
    # 以及测试图像存在于指定路径
    
    # 取消注释以运行相应示例
    
    # example_basic_inspection()
    # example_with_vector_store()
    # example_langgraph_workflow()
    # example_batch_processing()
    # example_advanced_retrieval()
    
    print("\n请取消注释相应的示例函数来运行测试")
    print("确保模型和测试数据已准备就绪")

