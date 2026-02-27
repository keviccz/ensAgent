import os
import uuid
import shutil
from datetime import datetime
from typing import List, Dict, Optional
from werkzeug.utils import secure_filename
from PIL import Image
from config import Config

class ImageManager:
    """图片管理器，负责图片的存储、管理和操作"""
    
    def __init__(self):
        """初始化图片管理器"""
        self.upload_folder = Config.UPLOAD_FOLDER
        self.allowed_extensions = Config.ALLOWED_EXTENSIONS
        self.max_file_size = Config.MAX_CONTENT_LENGTH
        
        # 确保上传文件夹存在
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def allowed_file(self, filename: str) -> bool:
        """检查文件扩展名是否允许"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """生成唯一的文件名"""
        # 获取文件扩展名
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        
        # 生成唯一ID
        unique_id = str(uuid.uuid4())
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 组合文件名
        return f"{timestamp}_{unique_id}.{file_extension}"
    
    def save_image(self, file, custom_filename: str = None) -> Dict[str, str]:
        """保存上传的图片"""
        try:
            # 检查文件是否存在
            if not file or file.filename == '':
                raise ValueError("没有选择文件")
            
            # 检查文件扩展名
            if not self.allowed_file(file.filename):
                raise ValueError(f"不支持的文件类型。支持的类型: {', '.join(self.allowed_extensions)}")
            
            # 生成文件名
            if custom_filename:
                filename = secure_filename(custom_filename)
            else:
                filename = self.generate_unique_filename(file.filename)
            
            # 构建完整路径
            file_path = os.path.join(self.upload_folder, filename)
            
            # 保存文件
            file.save(file_path)
            
            # 验证图片文件
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                # 如果图片验证失败，删除文件
                os.remove(file_path)
                raise ValueError("无效的图片文件")
            
            # 获取文件信息
            file_info = self.get_image_info(file_path)
            
            return {
                'filename': filename,
                'file_path': file_path,
                'size': file_info['size'],
                'dimensions': file_info['dimensions'],
                'upload_time': file_info['upload_time']
            }
            
        except Exception as e:
            raise Exception(f"保存图片失败: {str(e)}")
    
    def get_image_info(self, file_path: str) -> Dict[str, any]:
        """获取图片信息"""
        try:
            stat = os.stat(file_path)
            size = stat.st_size
            
            with Image.open(file_path) as img:
                dimensions = f"{img.width}x{img.height}"
            
            upload_time = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                'size': size,
                'dimensions': dimensions,
                'upload_time': upload_time
            }
        except Exception as e:
            return {
                'size': 0,
                'dimensions': 'Unknown',
                'upload_time': 'Unknown'
            }
    
    def list_images(self) -> List[Dict[str, str]]:
        """列出所有图片文件"""
        try:
            images = []
            
            for filename in os.listdir(self.upload_folder):
                file_path = os.path.join(self.upload_folder, filename)
                
                # 检查是否为文件且为支持的图片格式
                if os.path.isfile(file_path) and self.allowed_file(filename):
                    file_info = self.get_image_info(file_path)
                    
                    images.append({
                        'filename': filename,
                        'file_path': file_path,
                        'size': file_info['size'],
                        'dimensions': file_info['dimensions'],
                        'upload_time': file_info['upload_time']
                    })
            
            # 按上传时间排序（最新的在前）
            images.sort(key=lambda x: x['upload_time'], reverse=True)
            
            return images
            
        except Exception as e:
            raise Exception(f"获取图片列表失败: {str(e)}")
    
    def delete_image(self, filename: str) -> bool:
        """删除指定的图片"""
        try:
            file_path = os.path.join(self.upload_folder, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            else:
                return False
                
        except Exception as e:
            raise Exception(f"删除图片失败: {str(e)}")
    
    def get_image_path(self, filename: str) -> Optional[str]:
        """获取图片的完整路径"""
        file_path = os.path.join(self.upload_folder, filename)
        
        if os.path.exists(file_path) and self.allowed_file(filename):
            return file_path
        
        return None
    
    def resize_image(self, file_path: str, max_size: tuple = (1024, 1024)) -> str:
        """调整图片大小"""
        try:
            with Image.open(file_path) as img:
                # 计算新的尺寸
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 生成新文件名
                base_name = os.path.splitext(file_path)[0]
                extension = os.path.splitext(file_path)[1]
                resized_path = f"{base_name}_resized{extension}"
                
                # 保存调整后的图片
                img.save(resized_path, optimize=True, quality=85)
                
                return resized_path
                
        except Exception as e:
            raise Exception(f"调整图片大小失败: {str(e)}")
    
    def cleanup_old_images(self, days: int = 30) -> int:
        """清理指定天数前的图片"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days * 24 * 60 * 60)
            
            deleted_count = 0
            
            for filename in os.listdir(self.upload_folder):
                file_path = os.path.join(self.upload_folder, filename)
                
                if os.path.isfile(file_path):
                    file_time = os.path.getctime(file_path)
                    
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            raise Exception(f"清理旧图片失败: {str(e)}")

