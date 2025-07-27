（交流可以用英文，本文档中文，保留这句）

# 扩散模型教程项目说明

## 项目目标
编写一份 扩散模型设计的由浅入深的教程markdown，要包含大量的习题和参考答案（答案默认折叠）。合适时提及相关 pytorch 函数名但不写代码。
项目特色是，包含大量的可继续研究的线索

## 工具说明
当需要时，可以通过 `gemini -p "深入回答：<要问的问题> -m gemini-2.5-pro"` 来获取 gemini-2.5-pro 的参考意见(gemini 系只问 gemini-2.5-pro 不问别人)
当需要时，可以通过 `echo "<要问的问题>"|llm -m 4.1 来获取 gpt-4.1 的参考意见

## 教程大纲

### 最终章节结构（14章 + 附录）

1. **第1章：扩散模型导论** - 基本概念、历史发展、前向扩散过程
2. **第2章：神经网络架构：U-Net与ViT** - 扩散模型中的去噪网络架构，U-Net详解，Vision Transformer在扩散模型中的应用
3. **第3章：去噪扩散概率模型 (DDPM)** - 核心原理、变分下界、训练算法
4. **第4章：基于分数的生成模型** - Score matching、Langevin dynamics
5. **第5章：连续时间扩散模型 (PDE/SDE)** - 随机微分方程、概率流ODE、Fokker-Planck方程
6. **第6章：流匹配 (Flow Matching)** - 连续正则化流、最优传输视角、与扩散模型的联系
7. **第7章：扩散Transformer (DiT)** - Diffusion Transformer架构、与U-Net的对比、可扩展性分析
8. **第8章：采样算法与加速技术** - DDIM、DPM-Solver等快速采样方法
9. **第9章：条件生成与引导技术** - Classifier guidance、classifier-free guidance
10. **第10章：潜在扩散模型 (LDM)** - Stable Diffusion架构
11. **第11章：视频扩散模型** - 时序建模、3D U-Net、视频生成的挑战与方法
12. **第12章：文本扩散模型** - D3PM、Diffusion-LM、embedding空间扩散
13. **第13章：扩散模型的应用** - 图像生成、编辑、超分辨率、3D生成
14. **第14章：前沿研究与未来方向** - 一致性模型、扩散模型的未来发展趋势

**附录A：测度论与随机过程速成** - 为第5章PDE/SDE内容提供数学基础
**附录B：倒向随机微分方程 (BSDE) 速成** - 理解扩散模型反向过程的数学工具

### 内容设计原则

1. **PDE/SDE章节方法**：
   - 先介绍直觉和实际实现
   - 然后进行完整推导，包括reverse SDE
   - 测度论和随机微积分速成课程放在附录

2. **文本扩散模型重点**：
   - 离散状态空间扩散（如D3PM）
   - 连续embedding空间扩散（如Diffusion-LM）
   - 两者并重

3. **交互元素**：
   - 保持简单，先用静态图像
   - 逐步增加交互性

4. **编程语言和框架**：
   - Python/PyTorch（不用JAX）
   - 方法可以高级，但限于toy data

5. **章节依赖性**：
   - 每章尽量自包含
   - 文本扩散模型章节（第8章）设计为独立可读

6. **练习设计**：
   - 理论和实现混合
   - 包含挑战题
   - 难度递进

7. **代码框架**：
   - 逐章构建mini-library
   - 提供skeleton code让学生填充

8. **前置知识**：
   - 假设学生已有概率论、神经网络基础、PyTorch经验
   - 在首页明确说明这些前置要求

## 章节格式要求

每个章节应包含：

1. **开篇段落** - 引入本章主题，说明学习目标
2. **丰富的文字描述** - 不仅是公式，要有充分的文字解释和直观说明
3. **本章小结** - 总结要点，预告下一章内容

## 输出大小控制

**重要原则**：
- 输入可以是章节级别的请求（如"创建第2章"）
- 但输出必须限制在一个小节（section）的大小，不超过
- 有时甚至要在子小节（subsection）级别工作
- 这样确保每次生成的内容精炼且高质量

### 统一样式要求

1. **使用共享CSS/JS文件** - 将通用样式抽取到 `common.css` 和 `common.js`
2. **长代码和练习答案默认折叠** - 使用统一的折叠/展开机制
3. **响应式设计** - 确保移动端友好
4. **数学公式** - 使用KaTeX渲染
5. **代码高亮** - 使用Prism.js或类似库

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
