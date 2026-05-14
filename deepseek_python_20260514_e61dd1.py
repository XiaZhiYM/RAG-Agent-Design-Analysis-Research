"""
项目名称：基于 RAG 与多 Agent 的竞品洞察与设计前期调研系统
作者：产品设计师
核心能力：多 Agent 协作 + 长链推理 + RAG 检索
运行环境：Python 3.10+ (无需 API Key，使用模拟数据)
"""

import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# ------------------------------
# 1. 基础数据模型（结构化设计）
# ------------------------------

class DesignDimension(Enum):
    """设计分析维度"""
    FUNCTIONALITY = "功能体验"
    VISUAL = "视觉设计"
    INTERACTION = "交互流程"
    CONTENT = "内容策略"
    ACCESSIBILITY = "可访问性"

@dataclass
class RawMaterial:
    """原始资料数据模型"""
    source: str          # 来源：竞品官网 | App Store | 设计社区 | 行业报告
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DesignInsight:
    """设计洞察数据模型"""
    dimension: DesignDimension
    competitor: str
    finding: str
    sentiment: str        # positive | negative | neutral
    opportunity_score: float  # 0-10 分，代表设计机会优先级

@dataclass
class ResearchReport:
    """最终调研简报"""
    title: str
    competitors: List[str]
    pain_points: List[str]
    opportunity_points: List[DesignInsight]
    design_hypotheses: List[str]
    summary: str


# ------------------------------
# 2. RAG 向量检索系统（模拟实现）
# ------------------------------

class VectorStore:
    """
    模拟向量数据库（RAG 核心组件）
    真实场景中使用 FAISS / Chroma / Pinecone
    """
    def __init__(self):
        self.chunks = []
        self.index = {}
        self.chunk_id = 0
    
    def add_documents(self, documents: List[RawMaterial]):
        """将原始资料分块并存入向量库"""
        for doc in documents:
            # 模拟文本分块（真实场景使用语义分块）
            chunks = self._split_text(doc.content, max_chunk_size=300)
            for chunk in chunks:
                chunk_id = f"chunk_{self.chunk_id}"
                self.chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "source": doc.source,
                    "title": doc.title,
                    "metadata": doc.metadata
                })
                # 模拟向量嵌入（真实场景使用 OpenAI/DeepSeek 等 Embedding）
                self.index[chunk_id] = self._simulate_embedding(chunk)
                self.chunk_id += 1
        print(f"[RAG] ✅ 已存入 {len(documents)} 份资料，共 {self.chunk_id} 个文本块")
    
    def _split_text(self, text: str, max_chunk_size: int = 300) -> List[str]:
        """简单分块（实际用 RecursiveCharacterTextSplitter）"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        for word in words:
            if current_size + len(word) + 1 > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            current_chunk.append(word)
            current_size += len(word) + 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    
    def _simulate_embedding(self, text: str) -> List[float]:
        """模拟向量（真实场景调用 Embedding API）"""
        import hashlib
        import random
        # 基于文本内容的确定性随机向量
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        random.seed(hash_val)
        return [random.random() for _ in range(128)]
    
    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        根据查询检索相关文本块
        真实场景：计算 query 与 index 的余弦相似度
        """
        # 模拟检索：简单关键词匹配
        query_keywords = set(query.lower().split())
        scored_chunks = []
        for chunk_id, chunk_data in enumerate(self.chunks):
            chunk_text = chunk_data["text"].lower()
            # 计算关键词交集占比作为模拟相似度
            chunk_keywords = set(chunk_text.split())
            overlap = len(query_keywords & chunk_keywords)
            if overlap > 0:
                scored_chunks.append((chunk_id, overlap))
        # 按得分排序取前 k 个
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_k = [self.chunks[i]["text"] for i, _ in scored_chunks[:k]]
        return top_k if top_k else ["未找到相关材料，请扩大调研范围。"]


# ------------------------------
# 3. 多 Agent 系统（核心智能体）
# ------------------------------

class AgentBase:
    """所有 Agent 的基类"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
    
    def _call_llm(self, prompt: str) -> str:
        """
        模拟 LLM 调用（真实场景调用 OpenAI/DeepSeek 等接口）
        使用预置策略返回设计合理的内容
        """
        # 这里实现“模拟智能”，实际会替换为 LLM API
        return self._simulate_response(prompt)
    
    def _simulate_response(self, prompt: str) -> str:
        """模拟 LLM 回复（仅演示用，不依赖真实 API）"""
        # 实际运行时，这里替换为：
        # response = openai.ChatCompletion.create(...)
        # return response.choices[0].message.content
        
        # 预置策略：根据关键词返回设计相关话术
        prompt_lower = prompt.lower()
        if "竞品" in prompt_lower or "对比" in prompt_lower:
            return "分析了三个竞品的核心功能：A 侧重快捷支付，B 侧重社交分享，C 侧重数据可视化。建议融合 A 的效率与 B 的社交属性。"
        elif "用户评论" in prompt_lower or "差评" in prompt_lower:
            return "用户评论中提到的主要痛点：1. 注册流程复杂（12条）；2. 夜间模式缺失（8条）；3. 加载速度慢（7条）。这些是高频次、强情绪的关键设计机会点。"
        elif "设计趋势" in prompt_lower:
            return "近期设计趋势：微交互反馈（占比35%）、暗黑模式（28%）、圆角化UI（20%）。建议在本次设计中优先采用微交互方案，提升用户操作愉悦感。"
        elif "机会" in prompt_lower or "洞察" in prompt_lower:
            return "我发现了两个高价值设计机会：1. 用户普遍期望快捷操作入口，但竞品未做深度优化；2. 移动端表单填写体验普遍较差，可加入智能补全与实时校验。"
        else:
            return "经过综合推理，建议在功能层面增加 X 特性，在视觉层面优化 Y 组件，在交互层面改进 Z 流程。这将显著提升用户体验满意度。"


class CollectorAgent(AgentBase):
    """信息采集 Agent：负责获取原始资料"""
    def __init__(self):
        super().__init__("Collector", "信息采集与获取")
    
    def collect(self, keywords: List[str], sources: List[str]) -> List[RawMaterial]:
        print(f"\n🤖 [{self.name}] 开始采集：关键词={keywords}, 来源={sources}")
        
        # 模拟数据（真实场景：爬虫 / API）
        mock_data = {
            "竞品官网": [
                RawMaterial(
                    source="竞品官网",
                    title="竞品 A - 产品功能介绍页",
                    content="竞品 A 提供了快捷支付、账单管理、自动记账三大核心功能。其交互流程为：首页 -> 点击支付 -> 选择金额 -> 确认支付。用户反馈支付成功率达 98%。",
                    metadata={"competitor": "A", "category": "功能"}
                ),
                RawMaterial(
                    source="竞品官网",
                    title="竞品 B - 设计规范与组件",
                    content="竞品 B 采用了圆角设计系统（radius=12px），主色调为 #0066FF，使用卡片式布局。辅助功能包括深色模式和语音搜索。",
                    metadata={"competitor": "B", "category": "视觉"}
                )
            ],
            "App Store": [
                RawMaterial(
                    source="App Store",
                    title="竞品 A - 用户评论摘要（近30天）",
                    content="★★★★☆ 4.2分，1300+评论。主要差评：注册流程太复杂（20条）、没有夜间模式（15条）、通知推送过多（12条）。好评：支付速度快（50条）、界面简洁（45条）。",
                    metadata={"competitor": "A", "category": "用户反馈"}
                )
            ],
            "设计社区": [
                RawMaterial(
                    source="Dribbble",
                    title="2025年金融服务设计趋势",
                    content="趋势1：微交互与反馈动效（占趋势图表的45%）。趋势2：无障碍设计（WCAG 2.1 AAA 级，占比30%）。趋势3：个性化界面（占比25%）。建议优先采纳微交互趋势。",
                    metadata={"category": "趋势"}
                )
            ]
        }
        
        collected = []
        for source in sources:
            if source in mock_data:
                collected.extend(mock_data[source])
        
        time.sleep(0.5)  # 模拟处理延迟
        print(f"[{self.name}] 采集完成，共 {len(collected)} 份资料")
        return collected


class CleanerAgent(AgentBase):
    """资料清洗与分层 Agent：进行数据预处理和结构化"""
    def __init__(self, vector_store: VectorStore):
        super().__init__("Cleaner", "资料清洗与分层")
        self.vector_store = vector_store
    
    def process(self, raw_materials: List[RawMaterial]) -> List[RawMaterial]:
        print(f"\n🤖 [{self.name}] 开始清洗与分层")
        
        # 1. 去重（基于内容哈希）
        seen = set()
        unique_materials = []
        for material in raw_materials:
            content_hash = hash(material.content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_materials.append(material)
        
        # 2. 自动标注维度（模拟分类）
        for material in unique_materials:
            content_lower = material.content.lower()
            if "功能" in content_lower or "支付" in content_lower or "流程" in content_lower:
                material.metadata["dimension"] = DesignDimension.FUNCTIONALITY.value
            elif "视觉" in content_lower or "色彩" in content_lower or "字体" in content_lower:
                material.metadata["dimension"] = DesignDimension.VISUAL.value
            elif "交互" in content_lower or "点击" in content_lower or "反馈" in content_lower:
                material.metadata["dimension"] = DesignDimension.INTERACTION.value
            elif "用户" in content_lower or "评论" in content_lower or "反馈" in content_lower:
                material.metadata["dimension"] = DesignDimension.CONTENT.value
            else:
                material.metadata["dimension"] = DesignDimension.ACCESSIBILITY.value
        
        # 3. 存入 RAG 向量库
        self.vector_store.add_documents(unique_materials)
        
        print(f"[{self.name}] 清洗完成，去重后保留 {len(unique_materials)} 份，已存入 RAG 库")
        return unique_materials


class InsightAgent(AgentBase):
    """洞察推理 Agent：长链推理核心"""
    def __init__(self, vector_store: VectorStore):
        super().__init__("Insight", "洞察推理与长链分析")
        self.vector_store = vector_store
    
    def reason(self, query: str, competitors: List[str]) -> List[DesignInsight]:
        print(f"\n🤖 [{self.name}] 开始长链推理：{query}")
        
        # 第一步：检索相关材料（RAG 检索）
        relevant_chunks = self.vector_store.similarity_search(query, k=5)
        print(f"  检索到 {len(relevant_chunks)} 个相关文本块")
        
        # 第二步：构建长链推理过程（多步思考）
        # 模拟多步思维链（Chain of Thought）
        step1 = f"分析竞品在 {query} 方面的现状..."
        step2 = "识别用户评论中的高频关键词和情绪倾向..."
        step3 = "对比多个竞品的差异化设计策略..."
        step4 = "结合设计趋势，推导出高优先级的机会点..."
        
        full_reasoning = f"[推理步骤]\n{step1}\n{step2}\n{step3}\n{step4}\n"
        
        # 第三步：调用“模拟 LLM”生成洞察
        prompt = f"""
        请根据以下检索材料，生成 3 条高质量的设计洞察。
        每条洞察需包含：对应维度、竞品名称、具体发现、情感倾向、机会评分（0-10）。
        检索材料：{json.dumps(relevant_chunks, ensure_ascii=False)}
        查询主题：{query}
        竞品列表：{competitors}
        """
        
        # 模拟 LLM 生成结果（真实场景调用 API）
        insights = [
            DesignInsight(
                dimension=DesignDimension.FUNCTIONALITY,
                competitor="竞品 A",
                finding="用户普遍希望简化注册流程，但竞品 A 的流程超过 5 步，导致流失率高。",
                sentiment="negative",
                opportunity_score=9.2
            ),
            DesignInsight(
                dimension=DesignDimension.INTERACTION,
                competitor="竞品 B",
                finding="竞品 B 在支付反馈环节使用了微动效，用户满意度提升明显，但竞品 A 缺乏该设计。",
                sentiment="positive",
                opportunity_score=8.7
            ),
            DesignInsight(
                dimension=DesignDimension.VISUAL,
                competitor="竞品 C",
                finding="所有竞品均未支持深色模式，而用户评论中多次提及该需求，这是一个未被满足的市场空白。",
                sentiment="neutral",
                opportunity_score=9.5
            )
        ]
        
        print(f"[{self.name}] 推理完成，生成了 {len(insights)} 条设计洞察")
        return insights


class ReportAgent(AgentBase):
    """报告生成 Agent：最终输出结构化简报"""
    def __init__(self):
        super().__init__("Reporter", "调研简报生成")
    
    def generate_report(self, 
                       competitors: List[str], 
                       insights: List[DesignInsight],
                       pain_points: List[str]) -> ResearchReport:
        print(f"\n🤖 [{self.name}] 开始生成调研简报")
        
        # 基于洞察生成设计假设
        hypotheses = [
            f"通过简化注册流程至 2 步，预计可提升转化率 25%（基于洞察：{insights[0].finding}）",
            f"在支付环节加入微动效反馈，预计可提升用户操作满意度 30%（基于洞察：{insights[1].finding}）",
            f"优先开发深色模式，以覆盖 30% 的潜在用户群体（基于洞察：{insights[2].finding}）"
        ]
        
        # 生成摘要
        summary = f"""
        基于对 {', '.join(competitors)} 的深度竞品洞察，我们发现了 {len(pain_points)} 个关键痛点 
        和 {len(insights)} 个高价值设计机会。建议优先执行“简化注册流程”和“引入微交互反馈” 
        两个核心设计策略，预计可带来 25%-30% 的用户体验提升。
        """
        
        report = ResearchReport(
            title="竞品洞察与设计前期调研简报 (2025年Q2)",
            competitors=competitors,
            pain_points=pain_points,
            opportunity_points=insights,
            design_hypotheses=hypotheses,
            summary=summary
        )
        
        print(f"[{self.name}] ✅ 报告生成完成")
        return report


# ------------------------------
# 4. 主控制器（编排整个系统）
# ------------------------------

class DesignResearchOrchestrator:
    """
    总控 Agent：调度所有子 Agent，形成完整的工作流
    """
    def __init__(self):
        self.vector_store = VectorStore()
        self.collector = CollectorAgent()
        self.cleaner = CleanerAgent(self.vector_store)
        self.insight_engine = InsightAgent(self.vector_store)
        self.report_generator = ReportAgent()
    
    def run_full_research(self, 
                          keywords: List[str], 
                          sources: List[str],
                          query: str,
                          competitors: List[str]) -> ResearchReport:
        """
        全流程执行：采集 -> 清洗 -> 推理 -> 报告
        """
        print("=" * 60)
        print("🚀 启动：基于 RAG 与多 Agent 的竞品洞察系统")
        print("=" * 60)
        
        # Step 1: 信息采集（多 Agent 并行）
        raw_materials = self.collector.collect(keywords, sources)
        
        # Step 2: 资料清洗与分层（RAG 入库）
        clean_materials = self.cleaner.process(raw_materials)
        
        # Step 3: 洞察推理（长链推理）
        pain_points = ["注册流程复杂", "缺乏夜间模式", "支付反馈弱", "功能入口层级深"]
        insights = self.insight_engine.reason(query, competitors)
        
        # Step 4: 报告生成
        report = self.report_generator.generate_report(
            competitors, 
            insights, 
            pain_points
        )
        
        # 统计 Token 消耗（模拟）
        token_usage = self._estimate_token_usage(raw_materials, clean_materials, insights)
        print(f"\n📊 [Token 消耗统计]")
        print(f"   原始文本处理: {token_usage['raw']:,} tokens")
        print(f"   向量化存储: {token_usage['embedding']:,} tokens")
        print(f"   长链推理生成: {token_usage['inference']:,} tokens")
        print(f"   -----------------------------------")
        print(f"   总消耗: {token_usage['total']:,} tokens")
        
        return report
    
    def _estimate_token_usage(self, 
                             raw: List[RawMaterial], 
                             clean: List[RawMaterial], 
                             insights: List[DesignInsight]) -> Dict[str, int]:
        """模拟估算 Token 消耗（真实场景从 API 获取）"""
        total_text = sum(len(m.content) for m in raw) + sum(len(m.content) for m in clean)
        # 按中文字符估算：1 token ≈ 1.5 个汉字
        raw_tokens = int(total_text / 1.5)
        embedding_tokens = raw_tokens * 2  # 分块 + 嵌入开销
        inference_tokens = 3000  # 推理生成的逻辑链和结构化输出
        return {
            "raw": raw_tokens,
            "embedding": embedding_tokens,
            "inference": inference_tokens,
            "total": raw_tokens + embedding_tokens + inference_tokens
        }


# ------------------------------
# 5. 运行示例（一键执行）
# ------------------------------

if __name__ == "__main__":
    # 初始化总控
    orchestrator = DesignResearchOrchestrator()
    
    # 模拟用户输入
    research_query = "金融支付产品的用户注册与支付流程体验优化"
    target_competitors = ["竞品 A (PayFast)", "竞品 B (MoneyFlow)", "竞品 C (WealthHub)"]
    research_keywords = ["支付体验", "注册流程", "竞品对比", "设计趋势"]
    data_sources = ["竞品官网", "App Store", "设计社区", "行业报告"]
    
    # 运行全流程调研
    final_report = orchestrator.run_full_research(
        keywords=research_keywords,
        sources=data_sources,
        query=research_query,
        competitors=target_competitors
    )
    
    # 输出最终报告
    print("\n" + "=" * 60)
    print("📄 最终调研简报")
    print("=" * 60)
    print(f"📌 标题: {final_report.title}")
    print(f"📊 竞品: {', '.join(final_report.competitors)}")
    print(f"⚠️  关键痛点: {', '.join(final_report.pain_points)}")
    print(f"\n💡 设计机会点:")
    for i, insight in enumerate(final_report.opportunity_points, 1):
        print(f"  {i}. [{insight.dimension.value}] {insight.competitor}: {insight.finding}")
        print(f"     机会评分: {insight.opportunity_score}/10, 情感: {insight.sentiment}")
    
    print(f"\n🔍 设计假设:")
    for i, hyp in enumerate(final_report.design_hypotheses, 1):
        print(f"  {i}. {hyp}")
    
    print(f"\n📝 摘要:\n{final_report.summary}")
    print("\n✅ 系统执行完毕！")