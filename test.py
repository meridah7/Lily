import os
import requests
from PIL import Image
import uuid

def download_and_save_image(image_url: str, save_dir: str = "Data/Generated") -> str:
    """
    从远程 URL 下载图像，并保存到指定目录下，返回保存的文件路径。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 下载图像数据
    response = requests.get(image_url)
    response.raise_for_status()  # 如果请求失败，会抛出异常
    
    # 生成唯一的文件名
    filename = f"image_{uuid.uuid4().hex}.png"
    file_path = os.path.join(save_dir, filename)
    
    # 将下载的数据写入文件
    with open(file_path, "wb") as f:
        f.write(response.content)
    
    return file_path

def read_image(file_path: str) -> Image.Image:
    """
    从本地文件路径读取图像并返回 PIL Image 对象。
    """
    image = Image.open(file_path)
    return image

if __name__ == "__main__":
    # 示例远程图像 URL，请替换为实际 URL
    image_url = "https://cdn.leonardo.ai/users/534d9a32-be85-4c44-b7aa-0c44565d5320/generations/3eb93ce8-b6b2-4659-804c-cfa821a642ba/segments/1:1:1/Flux_Dev__with_the_text_FLUX_0.jpeg"
    
    try:
        # 下载并保存图像
        file_path = download_and_save_image(image_url)
        print(f"图像已保存到: {file_path}")
        
        # 读取图像
        image = read_image(file_path)
        print("图像读取成功，尺寸：", image.size)
    except Exception as e:
        print("处理图像时出错：", e)
