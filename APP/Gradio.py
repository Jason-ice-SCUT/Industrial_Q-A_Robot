import gradio as gr
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 在导入任何 huggingface 相关模块之前，强制设置镜像配置
# 使用国内镜像 hf-mirror.com
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'

# 导入镜像配置模块（确保缓存目录等设置生效）
import os_mirror.os_mirror

# 在线模式：明确设置允许网络访问
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'

# 强制配置 huggingface_hub 使用镜像（如果已安装）
try:
    import huggingface_hub
    # 直接设置端点常量
    huggingface_hub.constants.ENDPOINT = 'https://hf-mirror.com'
    # 设置文件下载端点
    if hasattr(huggingface_hub.constants, 'HF_HUB_ENDPOINT'):
        huggingface_hub.constants.HF_HUB_ENDPOINT = 'https://hf-mirror.com'
    
    # Monkey patch: 替换 URL 中的 huggingface.co 为 hf-mirror.com
    original_hf_hub_url = huggingface_hub.file_download.hf_hub_url
    def patched_hf_hub_url(*args, **kwargs):
        url = original_hf_hub_url(*args, **kwargs)
        if url and 'huggingface.co' in url:
            url = url.replace('huggingface.co', 'hf-mirror.com')
        return url
    huggingface_hub.file_download.hf_hub_url = patched_hf_hub_url
    
    print(f"已配置 huggingface_hub 使用镜像: {huggingface_hub.constants.ENDPOINT}")
except (ImportError, AttributeError) as e:
    print(f"配置 huggingface_hub 镜像时出错: {e}")
    pass

# 打印镜像配置信息（用于调试）
print(f"使用镜像站点: {os.environ.get('HUGGINGFACE_HUB_ENDPOINT', '未设置')}")

try:
    from RAG import query
    print("成功导入 query 模块")
except Exception as e:
    print(f"导入 query 模块失败: {e}")
    sys.exit(1)

title = "Industrial Q&A Robot (Online Version)"
description = "This is a chatbot that can answer questions based on the provided specific industrial context. (Online Version - can download models from Hugging Face if needed)"

QA_ROBOT = gr.Interface(
    fn = query.QA_Generate, 
    inputs = gr.Textbox(
        label="输入问题",
        placeholder="请输入您的问题...",
        lines=3,              # 初始显示3行
        max_lines=10,        # 最多可扩展到10行
        show_copy_button=True  # 显示复制按钮
    ),
    outputs = gr.Textbox(
        label="回答",
        lines=5,              # 初始显示5行
        max_lines=20,        # 最多可扩展到20行（输出通常更长）
        show_copy_button=True  # 显示复制按钮
    ),
    title = title, 
    description = description,
    examples = [
        ["电动平衡车的安全要求是什么？"],
        ["工业机器人在制造业中的应用有哪些？"],
        ["如何提高生产线的自动化水平？"]]
)

# Launch the interface
if __name__ == "__main__":
    print("=" * 50)
    print("启动 Industrial Q&A Robot - 在线版本")
    print("=" * 50)
    
    QA_ROBOT.launch()

