import sys
import os
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                            QFileDialog, QMessageBox, QInputDialog)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFont

# 导入各个模块
from data_manager import DataManager
from expert_selector import ExpertSelector
from ui_components import ExpertsTab, SelectionTab, ResultsTab, AvoidanceDialog

class ExpertSelectorApp(QMainWindow):
    """专家抽取助手主应用"""
    
    def __init__(self):
        super().__init__()
        # 初始化数据管理器和业务逻辑处理器
        self.data_manager = DataManager()
        self.selector = ExpertSelector(self.data_manager)
        
        # 初始化设置
        self.settings = QSettings("ExpertSelector", "Config")
        
        # 初始化UI
        self.init_ui()
        
        # 加载上次打开的文件
        self.load_last_file()
    
    def init_ui(self):
        """初始化界面"""
        # 设置窗口基本属性
        self.setWindowTitle("专家抽取助手")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        
        # 创建主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建顶部操作栏
        top_layout = QHBoxLayout()
        
        self.file_label = QLabel("未选择Excel文件")
        self.file_label.setFont(QFont("SimHei", 10))
        
        self.select_file_btn = QPushButton("选择Excel文件")
        self.select_file_btn.clicked.connect(self.select_excel_file)
        self.select_file_btn.setFont(QFont("SimHei", 10))
        self.select_file_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.refresh_btn = QPushButton("刷新数据")
        self.refresh_btn.clicked.connect(self.refresh_data)
        self.refresh_btn.setFont(QFont("SimHei", 10))
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        top_layout.addWidget(self.file_label)
        top_layout.addWidget(self.select_file_btn)
        top_layout.addWidget(self.refresh_btn)
        top_layout.addStretch()
        
        # 创建标签页控件
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("SimHei", 10))
        
        # 创建各个标签页组件
        self.experts_tab = ExpertsTab()
        self.selection_tab = SelectionTab()
        self.results_tab = ResultsTab()
        
        # 连接组件信号
        self.connect_signals()
        
        # 添加标签页到标签控件
        self.tabs.addTab(self.experts_tab, "专家库管理")
        self.tabs.addTab(self.selection_tab, "抽取设置")
        self.tabs.addTab(self.results_tab, "抽取结果")
        
        # 将顶部布局和标签控件添加到主布局
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.tabs)
    
    def connect_signals(self):
        """连接各个组件的信号和槽"""
        # 专家库标签页信号
        self.experts_tab.add_expert.connect(self.add_expert)
        self.experts_tab.edit_expert.connect(self.edit_expert)
        self.experts_tab.delete_expert.connect(self.delete_experts)
        self.experts_tab.set_search_callback(self.filter_experts)
        
        # 抽取设置标签页信号
        self.selection_tab.perform_selection.connect(self.perform_selection)
        
        # 结果标签页信号
        self.results_tab.export_results.connect(self.export_results)
        self.results_tab.save_record.connect(self.save_selection_record)
        self.results_tab.new_selection.connect(lambda: self.tabs.setCurrentIndex(1))
        self.results_tab.perform_avoidance.connect(self.handle_secondary_avoidance)  # 新增：连接规避信号
    
    def select_excel_file(self):
        """选择Excel文件并加载数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择专家库Excel文件", "", "Excel Files (*.xlsx *.xls)"
        )
        
        if file_path:
            self.load_excel_file(file_path)
    
    def load_last_file(self):
        """加载上次打开的文件"""
        last_file = self.settings.value("last_file", "")
        if last_file and os.path.exists(last_file):
            self.load_excel_file(last_file)
    
    def load_excel_file(self, file_path):
        """加载Excel文件"""
        try:
            self.data_manager.load_excel(file_path)
            self.file_label.setText(f"当前文件: {os.path.basename(file_path)}")
            
            # 更新专家表格
            self.update_experts_table()
            
            # 更新筛选控件
            directions = self.data_manager.get_directions()
            categories = self.data_manager.get_categories()
            self.selection_tab.update_filter_controls(
                self.data_manager.fields, directions, categories
            )
            
            # 启用相关按钮
            self.refresh_btn.setEnabled(True)
            self.selection_tab.set_enabled(True)
            
            # 保存最后打开的文件路径
            self.settings.setValue("last_file", file_path)
            
            QMessageBox.information(self, "成功", f"已加载 {len(self.data_manager.df)} 位专家数据")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件失败: {str(e)}")
    
    def refresh_data(self):
        """刷新数据"""
        if self.data_manager.excel_path:
            try:
                self.data_manager.load_excel(self.data_manager.excel_path)
                self.update_experts_table()
                
                # 更新筛选控件
                directions = self.data_manager.get_directions()
                categories = self.data_manager.get_categories()
                self.selection_tab.update_filter_controls(
                    self.data_manager.fields, directions, categories
                )
                
                QMessageBox.information(self, "成功", f"已刷新数据，共 {len(self.data_manager.df)} 位专家")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"刷新数据失败: {str(e)}")
    
    def update_experts_table(self):
        """更新专家表格"""
        if self.data_manager.df is not None:
            # 将DataFrame转换为字典列表
            data = self.data_manager.df.to_dict('records')
            self.experts_tab.update_experts(data, self.data_manager.fields)
    
    def filter_experts(self):
        """筛选专家"""
        if self.data_manager.df is None:
            return
            
        search_text = self.experts_tab.get_search_text()
        if not search_text:
            self.update_experts_table()
            return
            
        # 过滤数据
        filtered_data = []
        for _, row in self.data_manager.df.iterrows():
            if any(search_text in str(value).lower() for value in row.values):
                filtered_data.append(row.to_dict())
        
        # 更新表格
        self.experts_tab.update_experts(filtered_data, self.data_manager.fields)
    
    def add_expert(self):
        """添加新专家"""
        try:
            # 创建新专家数据
            expert_data = {}
            for field in self.data_manager.fields:
                if field.lower() in ["序号", "编号", "id"]:
                    # 序号自动生成
                    expert_data[field] = len(self.data_manager.df) + 1
                    continue
                    
                value, ok = QInputDialog.getText(
                    self, f"输入{field}", f"请输入{field}（留空为None）:"
                )
                if not ok:  # 用户取消
                    return
                expert_data[field] = value if value else None
            
            # 添加专家
            success = self.data_manager.add_expert(expert_data)
            if success:
                QMessageBox.information(self, "成功", "专家添加成功")
                self.update_experts_table()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"添加专家失败: {str(e)}")
    
    def edit_expert(self, index):
        """编辑专家信息"""
        try:
            if self.data_manager.df is None or index < 0 or index >= len(self.data_manager.df):
                raise Exception("无效的专家索引")
                
            # 获取当前专家数据
            expert_data = self.data_manager.df.iloc[index].to_dict()
            
            # 编辑专家数据
            updated_data = {}
            for field in self.data_manager.fields:
                current_value = expert_data[field]
                value, ok = QInputDialog.getText(
                    self, f"编辑{field}", 
                    f"请输入{field}（当前值: {current_value}）:",
                    text=str(current_value) if current_value is not None else ""
                )
                if not ok:  # 用户取消
                    return
                updated_data[field] = value if value else None
            
            # 更新专家
            success = self.data_manager.update_expert(index, updated_data)
            if success:
                QMessageBox.information(self, "成功", "专家信息更新成功")
                self.update_experts_table()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"编辑专家失败: {str(e)}")
    
    def delete_experts(self, indices):
        """删除专家"""
        try:
            # 确认删除
            reply = QMessageBox.question(
                self, "确认删除", 
                f"确定要删除选中的 {len(indices)} 位专家吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                success = self.data_manager.delete_experts(indices)
                if success:
                    QMessageBox.information(self, "成功", f"已删除 {len(indices)} 位专家")
                    self.update_experts_table()
                    
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除专家失败: {str(e)}")
    
    def perform_selection(self, params):
        """执行专家抽取"""
        try:
            # 补充必要参数默认值（如果UI未传递）
            full_params = {
                "selection_reason": params.get("selection_reason", ""),
                "professional_directions": params.get("professional_directions", []),
                "avoid_units": params.get("avoid_units", []),
                "technical_ratio": params.get("technical_ratio", 5),
                "venture_capital_ratio": params.get("venture_capital_ratio", 5)
            }
            # 执行抽取
            results = self.selector.select_experts(params)
            
            # 生成抽取信息
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_text = self.selector.generate_selection_info(params, timestamp)
            
            # 显示结果
            self.results_tab.display_results(
                results, 
                self.selector.get_groups(),
                self.data_manager.fields,
                info_text
            )
            
            # 切换到结果标签页
            self.tabs.setCurrentIndex(2)
            
        except Warning as w:
            # 显示警告信息但继续执行
            QMessageBox.warning(self, "提示", str(w))
            # 重新尝试显示结果（如果已经有结果）
            results = self.selector.get_selection_results()
            if results is not None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info_text = self.selector.generate_selection_info(params, timestamp)
                self.results_tab.display_results(
                    results.to_dict('records'), 
                    self.selector.get_groups(),
                    self.data_manager.fields,
                    info_text
                )
                self.tabs.setCurrentIndex(2)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"抽取专家失败: {str(e)}")
    
    # 新增：处理二次规避的方法
    def handle_secondary_avoidance(self, settings):
        """处理二次规避"""
        try:
            # 执行二次规避
            success, message = self.selector.perform_secondary_avoidance(settings)
            
            if success:
                # 生成更新后的抽取信息
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 获取原参数
                params = self.selector.original_params or {
                    'selection_reason': self.selection_tab.get_selection_reason(),
                    'professional_directions': [],
                    'avoid_units': []
                }
                info_text = self.selector.generate_selection_info(params, timestamp)
                
                # 更新显示结果
                results = self.selector.get_selection_results()
                if results is not None:
                    self.results_tab.display_results(
                        results.to_dict('records'), 
                        self.selector.get_groups(),
                        self.data_manager.fields,
                        info_text
                    )
                
                QMessageBox.information(self, "成功", message)
            else:
                QMessageBox.information(self, "信息", message)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"执行规避失败: {str(e)}")
    
    def export_results(self):
        """导出抽取结果"""
        try:
            results = self.selector.get_selection_results()
            if results is None:
                raise Exception("没有抽取结果可导出")
                
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存抽取结果", 
                f"专家抽取结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", 
                "Excel Files (*.xlsx)"
            )
            
            if file_path:
                success = self.data_manager.export_results(results, file_path)
                if success:
                    QMessageBox.information(self, "成功", f"抽取结果已导出到:\n{file_path}")
                    
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def save_selection_record(self):
        """保存抽取记录"""
        try:
            results = self.selector.get_selection_results()
            if results is None:
                raise Exception("没有抽取结果可保存")
                
            info_text = self.results_tab.get_selection_info()
            sheet_name = self.data_manager.save_selection_record(results, info_text)
            
            QMessageBox.information(
                self, "成功", 
                f"抽取记录已保存到Excel文件的 '{sheet_name}' 工作表"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存记录失败: {str(e)}")


if __name__ == "__main__":
    # 确保中文显示正常
    app = QApplication(sys.argv)
    app.setFont(QFont("SimHei", 10))
    
    window = ExpertSelectorApp()
    window.show()
    
    sys.exit(app.exec())
