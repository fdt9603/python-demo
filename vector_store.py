"""
向量数据库模块
用于存储和检索历史PCB缺陷检测结果
支持相似缺陷案例检索、案例库管理
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("警告: ChromaDB未安装，将使用内存存储")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers未安装，向量嵌入将不可用")


class PCBVectorStore:
    """PCB缺陷检测结果向量存储"""
    
    def __init__(
        self,
        collection_name: str = "pcb_defects",
        persist_directory: str = "./vector_db",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_model: 嵌入模型名称
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # 初始化嵌入模型
        self.embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                print(f"加载嵌入模型: {embedding_model}")
            except Exception as e:
                print(f"加载嵌入模型失败: {e}")
        
        # 初始化向量数据库
        self.client = None
        self.collection = None
        
        if CHROMA_AVAILABLE:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "PCB缺陷检测结果存储"}
            )
            print(f"向量数据库已初始化: {persist_directory}")
        else:
            # 内存存储作为后备
            self.memory_store: List[Dict[str, Any]] = []
            print("使用内存存储（ChromaDB未安装）")
    
    def _encode_text(self, text: str) -> Optional[List[float]]:
        """将文本编码为向量"""
        if self.embedder is None:
            return None
        
        try:
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"编码失败: {e}")
            return None
    
    def _create_text_description(self, defect_data: Dict[str, Any]) -> str:
        """创建缺陷的文本描述用于嵌入"""
        defect_type = defect_data.get("defect", "unknown")
        bbox = defect_data.get("bbox", [])
        repair = defect_data.get("repair", "")
        
        # 创建描述文本
        description = f"缺陷类型: {defect_type}, "
        description += f"位置: {bbox}, "
        description += f"维修建议: {repair}"
        
        return description
    
    def add_detection_result(
        self,
        image_path: str,
        defects: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加检测结果到向量数据库
        
        Args:
            image_path: 图像路径
            defects: 缺陷列表
            metadata: 额外元数据
        
        Returns:
            文档ID
        """
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 准备元数据
        result_metadata = {
            "image_path": image_path,
            "defect_count": len(defects),
            "timestamp": datetime.now().isoformat(),
            "defects_json": json.dumps(defects, ensure_ascii=False)
        }
        
        if metadata:
            result_metadata.update(metadata)
        
        # 创建文档文本
        if defects:
            # 如果有多个缺陷，合并描述
            descriptions = [
                self._create_text_description(d) for d in defects
            ]
            document_text = " | ".join(descriptions)
        else:
            document_text = "正常，无缺陷"
        
        # 生成嵌入
        embedding = self._encode_text(document_text)
        
        if CHROMA_AVAILABLE and self.collection:
            # 使用ChromaDB
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding] if embedding else None,
                documents=[document_text],
                metadatas=[result_metadata]
            )
        else:
            # 内存存储
            self.memory_store.append({
                "id": doc_id,
                "text": document_text,
                "embedding": embedding,
                "metadata": result_metadata
            })
        
        print(f"已添加检测结果: {doc_id}")
        return doc_id
    
    def search_similar_defects(
        self,
        query_defects: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        搜索相似的缺陷案例
        
        Args:
            query_defects: 查询缺陷列表
            top_k: 返回最相似的k个结果
            min_score: 最小相似度分数
        
        Returns:
            相似案例列表，包含相似度和元数据
        """
        if not query_defects:
            return []
        
        # 创建查询文本
        query_texts = [self._create_text_description(d) for d in query_defects]
        query_text = " | ".join(query_texts)
        
        # 生成查询嵌入
        query_embedding = self._encode_text(query_text)
        
        if query_embedding is None:
            print("警告: 无法生成查询嵌入，返回空结果")
            return []
        
        results = []
        
        if CHROMA_AVAILABLE and self.collection:
            # 使用ChromaDB查询
            query_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            for i, doc_id in enumerate(query_results['ids'][0]):
                distance = query_results['distances'][0][i] if 'distances' in query_results else 0
                # 将距离转换为相似度分数（ChromaDB使用余弦距离）
                similarity = 1 - distance if distance else 0
                
                if similarity >= min_score:
                    metadata = query_results['metadatas'][0][i]
                    metadata['similarity'] = similarity
                    metadata['id'] = doc_id
                    results.append(metadata)
        else:
            # 内存存储查询
            if not self.memory_store:
                return []
            
            # 计算相似度
            similarities = []
            for item in self.memory_store:
                if item['embedding'] is None:
                    continue
                
                # 余弦相似度
                similarity = np.dot(
                    query_embedding,
                    item['embedding']
                ) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(item['embedding'])
                )
                
                if similarity >= min_score:
                    similarities.append({
                        'similarity': float(similarity),
                        'item': item
                    })
            
            # 排序并取前k个
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results = [
                {
                    **s['item']['metadata'],
                    'similarity': s['similarity'],
                    'id': s['item']['id']
                }
                for s in similarities[:top_k]
            ]
        
        return results
    
    def get_case_by_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取案例"""
        if CHROMA_AVAILABLE and self.collection:
            results = self.collection.get(ids=[case_id])
            if results['ids']:
                metadata = results['metadatas'][0]
                metadata['id'] = case_id
                return metadata
        else:
            for item in self.memory_store:
                if item['id'] == case_id:
                    return {**item['metadata'], 'id': item['id']}
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        if CHROMA_AVAILABLE and self.collection:
            count = self.collection.count()
            return {
                "total_cases": count,
                "storage_type": "ChromaDB",
                "persist_directory": self.persist_directory
            }
        else:
            return {
                "total_cases": len(self.memory_store),
                "storage_type": "memory",
                "persist_directory": None
            }
    
    def delete_case(self, case_id: str) -> bool:
        """删除案例"""
        if CHROMA_AVAILABLE and self.collection:
            try:
                self.collection.delete(ids=[case_id])
                print(f"已删除案例: {case_id}")
                return True
            except Exception as e:
                print(f"删除失败: {e}")
                return False
        else:
            self.memory_store = [item for item in self.memory_store if item['id'] != case_id]
            return True
    
    def export_to_json(self, output_path: str):
        """导出所有数据到JSON文件"""
        all_cases = []
        
        if CHROMA_AVAILABLE and self.collection:
            results = self.collection.get()
            for i, doc_id in enumerate(results['ids']):
                all_cases.append({
                    'id': doc_id,
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                })
        else:
            for item in self.memory_store:
                all_cases.append({
                    'id': item['id'],
                    'document': item['text'],
                    'metadata': item['metadata']
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_cases, f, ensure_ascii=False, indent=2)
        
        print(f"已导出 {len(all_cases)} 个案例到: {output_path}")


def create_vector_store(
    collection_name: str = "pcb_defects",
    persist_directory: str = "./vector_db"
) -> PCBVectorStore:
    """创建向量存储实例"""
    return PCBVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )


if __name__ == "__main__":
    # 测试向量存储
    print("测试向量存储...")
    
    store = create_vector_store()
    
    # 添加测试数据
    test_defects = [
        {
            "defect": "short",
            "bbox": [120, 350, 45, 12],
            "repair": "清理焊锡桥接"
        }
    ]
    
    doc_id = store.add_detection_result(
        image_path="test_board.jpg",
        defects=test_defects,
        metadata={"board_type": "test"}
    )
    
    # 搜索相似案例
    similar_cases = store.search_similar_defects(
        query_defects=test_defects,
        top_k=3
    )
    
    print(f"\n找到 {len(similar_cases)} 个相似案例:")
    for case in similar_cases:
        print(f"  相似度: {case['similarity']:.3f}, ID: {case['id']}")
    
    # 统计信息
    stats = store.get_statistics()
    print(f"\n数据库统计: {stats}")

