import pandas as pd
import os
from datetime import datetime

class DataManager:
    """数据管理类，处理Excel文件的加载、保存和编辑"""
    
    def __init__(self):
        self.df = None           # 专家数据DataFrame
        self.excel_path = None   # Excel文件路径
        self.fields = []         # 数据字段列表
    
    def load_excel(self, file_path):
        """加载Excel文件"""
        try:
            # 尝试不同的引擎读取Excel文件
            try:
                self.df = pd.read_excel(file_path, engine='openpyxl')
            except:
                self.df = pd.read_excel(file_path, engine='xlrd')
            
            self.excel_path = file_path
            self.fields = list(self.df.columns)
            return True
        except Exception as e:
            raise Exception(f"加载Excel文件失败: {str(e)}")
    
    def add_expert(self, expert_data):
        """添加新专家"""
        if self.df is None:
            raise Exception("请先加载专家库文件")
            
        try:
            # 创建新行
            new_row = pd.DataFrame([expert_data])
            # 添加到DataFrame
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            # 保存到文件
            self.save_excel()
            return True
        except Exception as e:
            raise Exception(f"添加专家失败: {str(e)}")
    
    def update_expert(self, index, expert_data):
        """更新专家信息"""
        if self.df is None:
            raise Exception("请先加载专家库文件")
            
        if index < 0 or index >= len(self.df):
            raise Exception("无效的专家索引")
            
        try:
            # 更新数据
            for key, value in expert_data.items():
                if key in self.df.columns:
                    self.df.at[index, key] = value
            # 保存到文件
            self.save_excel()
            return True
        except Exception as e:
            raise Exception(f"更新专家信息失败: {str(e)}")
    
    def delete_experts(self, indices):
        """删除专家"""
        if self.df is None:
            raise Exception("请先加载专家库文件")
            
        try:
            # 删除选中的行
            self.df = self.df.drop(indices).reset_index(drop=True)
            # 保存到文件
            self.save_excel()
            return True
        except Exception as e:
            raise Exception(f"删除专家失败: {str(e)}")
    
    def save_excel(self):
        """保存Excel文件"""
        if self.df is None or not self.excel_path:
            raise Exception("没有可保存的数据")
            
        try:
            # 保存到原文件
            self.df.to_excel(self.excel_path, index=False, engine='openpyxl')
            return True
        except Exception as e:
            raise Exception(f"保存Excel文件失败: {str(e)}")
    
    def export_results(self, results, file_path):
        """导出抽取结果到Excel文件"""
        try:
            results.to_excel(file_path, index=False, engine='openpyxl')
            return True
        except Exception as e:
            raise Exception(f"导出结果失败: {str(e)}")
    
    def save_selection_record(self, results, info_text):
        """保存抽取记录到原Excel文件的新工作表"""
        if not self.excel_path or results is None:
            raise Exception("没有可保存的抽取记录")
            
        try:
            # 创建新工作表名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sheet_name = f"抽取记录_{timestamp}"
            
            # 保存到现有Excel文件
            with pd.ExcelWriter(
                self.excel_path, 
                engine='openpyxl',
                mode='a',
                if_sheet_exists='new'
            ) as writer:
                # 先写入抽取信息
                info_df = pd.DataFrame([info_text], columns=["抽取信息"])
                info_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 在信息下方写入抽取结果
                results.to_excel(writer, sheet_name=sheet_name, startrow=3, index=False)
            
            return sheet_name
        except Exception as e:
            raise Exception(f"保存抽取记录失败: {str(e)}")
    
    def get_directions(self):
        """获取所有专业方向"""
        if self.df is None or '专业方向' not in self.df.columns:
            return []
            
        # 获取不重复的专业方向列表
        return sorted(self.df['专业方向'].dropna().unique().tolist())
    
    def get_categories(self):
        """获取所有专家类别"""
        if self.df is None or '类别' not in self.df.columns:
            return []
            
        # 获取不重复的类别列表
        return sorted(self.df['类别'].dropna().unique().tolist())
