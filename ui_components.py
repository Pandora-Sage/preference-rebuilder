from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                            QLineEdit, QTableWidget, QTableWidgetItem, QGroupBox,
                            QFormLayout, QSpinBox, QTextEdit, QComboBox, QScrollArea,
                            QHeaderView, QCheckBox, QRadioButton, QButtonGroup,
                            QTabWidget, QListWidget, QListWidgetItem)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtWidgets import QDialog


class ExpertsTab(QWidget):
    """专家库管理标签页"""
    
    add_expert = pyqtSignal()
    edit_expert = pyqtSignal(int)
    delete_expert = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.search_callback = None
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # 创建搜索框
        search_layout = QHBoxLayout()
        search_label = QLabel("搜索:")
        search_label.setFont(QFont("SimHei", 10))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词搜索专家...")
        self.search_input.setFont(QFont("SimHei", 10))
        self.search_input.textChanged.connect(self.on_search_changed)
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        search_layout.addStretch()
        
        # 创建表格
        self.experts_table = QTableWidget()
        self.experts_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.experts_table.setFont(QFont("SimHei", 10))
        self.experts_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f9f9f9;
                alternate-background-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #cccccc;
            }
        """)
        self.experts_table.setAlternatingRowColors(True)
        
        # 创建按钮布局
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("新增专家")
        self.add_btn.clicked.connect(self.add_expert.emit)
        
        self.edit_btn = QPushButton("编辑专家")
        self.edit_btn.clicked.connect(self.on_edit_clicked)
        self.edit_btn.setEnabled(False)
        
        self.delete_btn = QPushButton("删除专家")
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        self.delete_btn.setEnabled(False)
        
        for btn in [self.add_btn, self.edit_btn, self.delete_btn]:
            btn.setFont(QFont("SimHei", 10))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #555555;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 3px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #333333;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
            btn_layout.addWidget(btn)
        
        btn_layout.addStretch()
        
        # 添加到布局
        layout.addLayout(search_layout)
        layout.addWidget(self.experts_table)
        layout.addLayout(btn_layout)
        
        # 连接表格选择变化信号
        self.experts_table.itemSelectionChanged.connect(self.on_selection_changed)
    
    def on_selection_changed(self):
        """处理表格选择变化"""
        selected_rows = set(item.row() for item in self.experts_table.selectedItems())
        has_selection = len(selected_rows) > 0
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
    
    def on_edit_clicked(self):
        """处理编辑按钮点击"""
        selected_rows = set(item.row() for item in self.experts_table.selectedItems())
        if selected_rows:
            # 取第一个选中的行
            row = next(iter(selected_rows))
            self.edit_expert.emit(row)
    
    def on_delete_clicked(self):
        """处理删除按钮点击"""
        selected_rows = sorted(set(item.row() for item in self.experts_table.selectedItems()), reverse=True)
        if selected_rows:
            self.delete_expert.emit(selected_rows)
    
    def on_search_changed(self):
        """处理搜索文本变化"""
        if self.search_callback:
            self.search_callback()
    
    def set_search_callback(self, callback):
        """设置搜索回调函数"""
        self.search_callback = callback
    
    def get_search_text(self):
        """获取搜索文本"""
        return self.search_input.text().lower()
    
    def update_experts(self, experts_data, fields):
        """更新专家表格数据"""
        # 保存当前选择状态
        selected_rows = set(item.row() for item in self.experts_table.selectedItems())
        
        # 设置表格行数和列数
        self.experts_table.setRowCount(len(experts_data))
        self.experts_table.setColumnCount(len(fields))
        self.experts_table.setHorizontalHeaderLabels(fields)
        
        # 填充表格数据
        for row_idx, expert in enumerate(experts_data):
            for col_idx, field in enumerate(fields):
                value = expert.get(field, "")
                item = QTableWidgetItem(str(value) if value is not None else "")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.experts_table.setItem(row_idx, col_idx, item)
        
        # 恢复选择状态（如果有）
        for row in selected_rows:
            if row < len(experts_data):
                self.experts_table.selectRow(row)


class SelectionTab(QWidget):
    """抽取设置标签页，添加了规避功能和可输入的专业方向"""
    
    perform_selection = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """初始化界面，添加规避设置和可输入的专业方向"""
        layout = QVBoxLayout(self)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # 创建抽取事由设置
        reason_group = QGroupBox("抽取事由")
        reason_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        reason_layout = QVBoxLayout()
        
        self.reason_group = QButtonGroup(self)
        self.reason1_radio = QRadioButton("2025年第十四届中国创新创业大赛北斗应用专业赛初赛")
        self.reason2_radio = QRadioButton("2025年第十四届中国创新创业大赛北斗应用专业赛复赛")
        
        # 默认选择初赛
        self.reason1_radio.setChecked(True)
        
        for radio in [self.reason1_radio, self.reason2_radio]:
            radio.setFont(QFont("SimHei", 10))
            self.reason_group.addButton(radio)
            reason_layout.addWidget(radio)
        
        reason_group.setLayout(reason_layout)
        
        # 创建专业方向筛选（改为可输入）
        direction_group = QGroupBox("专业方向筛选")
        direction_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        direction_layout = QVBoxLayout()
        
        # 改为可编辑的ComboBox，允许输入
        self.direction_combo = QComboBox()
        self.direction_combo.setEditable(True)
        self.direction_combo.setFont(QFont("SimHei", 10))
        self.direction_combo.setMinimumHeight(30)
        
        # 添加"全选"选项
        self.select_all_directions = QCheckBox("全选所有专业方向")
        self.select_all_directions.setFont(QFont("SimHei", 10))
        self.select_all_directions.setChecked(True)
        self.select_all_directions.stateChanged.connect(self.on_select_all_directions)
        
        direction_layout.addWidget(QLabel("请选择或输入专业方向:"))
        direction_layout.addWidget(self.direction_combo)
        direction_layout.addWidget(self.select_all_directions)
        
        direction_group.setLayout(direction_layout)
        
        # 创建规避设置
        avoidance_group = QGroupBox("规避设置")
        avoidance_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        avoidance_layout = QVBoxLayout()
        
        # 规避单位输入
        self.avoid_unit_input = QLineEdit()
        self.avoid_unit_input.setPlaceholderText("输入需要规避的单位...")
        self.avoid_unit_input.setFont(QFont("SimHei", 10))
        
        # 添加规避按钮
        self.add_avoid_btn = QPushButton("添加到规避列表")
        self.add_avoid_btn.setFont(QFont("SimHei", 10))
        self.add_avoid_btn.clicked.connect(self.add_avoid_unit)
        
        # 规避列表
        avoidance_layout.addWidget(QLabel("需要规避的单位:"))
        avoidance_layout.addWidget(self.avoid_unit_input)
        avoidance_layout.addWidget(self.add_avoid_btn)
        
        self.avoid_list = QListWidget()
        self.avoid_list.setFont(QFont("SimHei", 10))
        avoidance_layout.addWidget(self.avoid_list)
        
        # 移除选中规避项按钮
        self.remove_avoid_btn = QPushButton("移除选中项")
        self.remove_avoid_btn.setFont(QFont("SimHei", 10))
        self.remove_avoid_btn.clicked.connect(self.remove_avoid_unit)
        avoidance_layout.addWidget(self.remove_avoid_btn)
        
        avoidance_group.setLayout(avoidance_layout)
        
        # 创建抽取比例设置
        ratio_group = QGroupBox("备选评委抽取比例")
        ratio_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        ratio_layout = QFormLayout()
        ratio_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        ratio_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        self.technical_ratio = QSpinBox()
        self.technical_ratio.setMinimum(1)
        self.technical_ratio.setMaximum(10)
        self.technical_ratio.setValue(5)
        self.technical_ratio.setFont(QFont("SimHei", 10))
        
        self.venture_ratio = QSpinBox()
        self.venture_ratio.setMinimum(1)
        self.venture_ratio.setMaximum(10)
        self.venture_ratio.setValue(5)
        self.venture_ratio.setFont(QFont("SimHei", 10))
        
        ratio_layout.addRow(QLabel("技术评委 (1:)"), self.technical_ratio)
        ratio_layout.addRow(QLabel("创投评委 (1:)"), self.venture_ratio)
        
        ratio_group.setLayout(ratio_layout)
        
        # 创建按钮
        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("开始抽取")
        self.select_btn.clicked.connect(self.on_perform_selection)
        self.select_btn.setEnabled(False)
        self.select_btn.setFont(QFont("SimHei", 10))
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.select_btn)
        
        # 添加到滚动布局
        scroll_layout.addWidget(reason_group)
        scroll_layout.addWidget(direction_group)
        scroll_layout.addWidget(avoidance_group)
        scroll_layout.addWidget(ratio_group)
        scroll_layout.addStretch()
        
        # 设置滚动区域
        scroll_area.setWidget(scroll_content)
        
        # 添加到主布局
        layout.addWidget(scroll_area)
        layout.addLayout(btn_layout)
        
        # 存储专业方向数据和规避单位
        self.all_directions = []
        self.avoid_units = []
    
    def add_avoid_unit(self):
        """添加单位到规避列表"""
        unit = self.avoid_unit_input.text().strip()
        if unit and unit not in self.avoid_units:
            self.avoid_units.append(unit)
            self.avoid_list.addItem(unit)
            self.avoid_unit_input.clear()
    
    def remove_avoid_unit(self):
        """从规避列表移除选中单位"""
        selected_items = self.avoid_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            unit = item.text()
            if unit in self.avoid_units:
                self.avoid_units.remove(unit)
            row = self.avoid_list.row(item)
            self.avoid_list.takeItem(row)
    
    def update_filter_controls(self, fields, directions=None, categories=None):
        """更新筛选控件"""
        # 保存专业方向数据
        if directions:
            self.all_directions = directions
            # 添加到下拉框但不选中
            self.direction_combo.clear()
            self.direction_combo.addItems(directions)
    
    def on_select_all_directions(self, state):
        """处理全选专业方向"""
        self.direction_combo.setEnabled(not state)
    
    def on_perform_selection(self):
        """执行抽取"""
        # 收集抽取参数
        params = {
            # 获取抽取事由
            "selection_reason": self.reason1_radio.text() if self.reason1_radio.isChecked() else self.reason2_radio.text(),
            
            # 获取专业方向筛选条件
            "professional_directions": self.all_directions if self.select_all_directions.isChecked() 
                                      else [self.direction_combo.currentText()] if self.direction_combo.currentText() 
                                      else [],
            
            # 获取规避单位列表
            "avoid_units": self.avoid_units,
            
            # 获取抽取比例
            "technical_ratio": self.technical_ratio.value(),
            "venture_capital_ratio": self.venture_ratio.value()
        }
        
        # 发送抽取信号
        self.perform_selection.emit(params)
    
    def set_enabled(self, enabled):
        """设置控件是否可用"""
        self.select_btn.setEnabled(enabled)
        self.reason1_radio.setEnabled(enabled)
        self.reason2_radio.setEnabled(enabled)
        self.direction_combo.setEnabled(enabled and not self.select_all_directions.isChecked())
        self.select_all_directions.setEnabled(enabled)
        self.avoid_unit_input.setEnabled(enabled)
        self.add_avoid_btn.setEnabled(enabled)
        self.remove_avoid_btn.setEnabled(enabled)
        self.technical_ratio.setEnabled(enabled)
        self.venture_ratio.setEnabled(enabled)
    
    def get_selection_reason(self):
        """获取抽取事由"""
        return self.reason1_radio.text() if self.reason1_radio.isChecked() else self.reason2_radio.text()


class AvoidanceDialog(QDialog):
    """规避设置弹窗"""
    def __init__(self, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置规避条件")
        self.setMinimumWidth(400)
        self.fields = fields
        self.selected_field = None
        self.avoid_values = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 字段选择
        field_group = QGroupBox("选择规避字段")
        field_layout = QVBoxLayout()
        self.field_combo = QComboBox()
        self.field_combo.addItems(self.fields)
        # 默认选择"单位"字段如果存在
        if "单位" in self.fields:
            self.field_combo.setCurrentText("单位")
        self.field_combo.currentTextChanged.connect(self.on_field_changed)
        field_layout.addWidget(self.field_combo)
        field_group.setLayout(field_layout)
        
        # 规避值设置
        value_group = QGroupBox("设置需要规避的值")
        value_layout = QVBoxLayout()
        
        value_input_layout = QHBoxLayout()
        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText(f"输入需要规避的{self.field_combo.currentText()}...")
        self.add_value_btn = QPushButton("添加")
        self.add_value_btn.clicked.connect(self.add_avoid_value)
        
        value_input_layout.addWidget(self.value_input)
        value_input_layout.addWidget(self.add_value_btn)
        
        self.value_list = QListWidget()
        self.remove_value_btn = QPushButton("移除选中项")
        self.remove_value_btn.clicked.connect(self.remove_avoid_value)
        
        value_layout.addLayout(value_input_layout)
        value_layout.addWidget(self.value_list)
        value_layout.addWidget(self.remove_value_btn)
        value_group.setLayout(value_layout)
        
        # 确认按钮
        btn_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("确认")
        self.confirm_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addWidget(field_group)
        layout.addWidget(value_group)
        layout.addLayout(btn_layout)
        
    def on_field_changed(self, text):
        self.value_input.setPlaceholderText(f"输入需要规避的{text}...")
        
    def add_avoid_value(self):
        value = self.value_input.text().strip()
        if value and value not in self.avoid_values:
            self.avoid_values.append(value)
            self.value_list.addItem(value)
            self.value_input.clear()
            
    def remove_avoid_value(self):
        selected_items = self.value_list.selectedItems()
        if not selected_items:
            return
            
        for item in selected_items:
            value = item.text()
            if value in self.avoid_values:
                self.avoid_values.remove(value)
            row = self.value_list.row(item)
            self.value_list.takeItem(row)
            
    def get_avoidance_settings(self):
        return {
            "field": self.field_combo.currentText(),
            "values": self.avoid_values
        }


class ResultsTab(QWidget):
    """抽取结果标签页"""
    
    export_results = pyqtSignal()
    save_record = pyqtSignal()
    new_selection = pyqtSignal()
    perform_avoidance = pyqtSignal(dict)  # 新增：规避信号
    
    def __init__(self):
        super().__init__()
        self.fields = []  # 保存字段信息
        self.all_candidates = []  # 保存所有备选专家
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        
        # 创建抽取信息区域
        info_group = QGroupBox("抽取信息")
        info_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("SimHei", 10))
        self.info_text.setStyleSheet("background-color: #f0f0f0;")
        
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        
        # 创建结果标签页控件
        self.results_tabs = QTabWidget()
        self.results_tabs.setFont(QFont("SimHei", 10))
        
        # 创建备选专家列表标签页
        self.candidates_table = QTableWidget()
        self.candidates_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.candidates_table.setFont(QFont("SimHei", 10))
        self.candidates_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #f9f9f9;
                alternate-background-color: #f0f0f0;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 4px;
                border: 1px solid #cccccc;
            }
        """)
        self.candidates_table.setAlternatingRowColors(True)
        
        self.results_tabs.addTab(self.candidates_table, "备选专家名单")
        
        # 创建按钮布局
        btn_layout = QHBoxLayout()
        
        # 新增：规避设置按钮
        self.avoidance_btn = QPushButton("规避设置")
        self.avoidance_btn.clicked.connect(self.on_avoidance_clicked)
        self.avoidance_btn.setEnabled(False)
        
        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results.emit)
        self.export_btn.setEnabled(False)
        
        self.save_btn = QPushButton("保存抽取记录")
        self.save_btn.clicked.connect(self.save_record.emit)
        self.save_btn.setEnabled(False)
        
        self.new_selection_btn = QPushButton("重新抽取")
        self.new_selection_btn.clicked.connect(self.new_selection.emit)
        self.new_selection_btn.setEnabled(False)
        
        for btn in [self.avoidance_btn, self.export_btn, self.save_btn, self.new_selection_btn]:
            btn.setFont(QFont("SimHei", 10))
            btn.setStyleSheet("""
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
            btn_layout.addWidget(btn)
        
        btn_layout.addStretch()
        
        # 添加到布局
        layout.addWidget(info_group)
        layout.addWidget(self.results_tabs)
        layout.addLayout(btn_layout)
        
        # 存储分组表格
        self.group_tables = {}
    
    def display_results(self, candidates, groups=None, fields=None, info_text=None):
        """显示抽取结果"""
        # 保存字段信息和备选专家
        self.fields = fields
        self.all_candidates = candidates.copy()
        
        # 清空现有分组标签页
        while self.results_tabs.count() > 1:  # 保留备选专家名单标签页
            self.results_tabs.removeTab(1)
        self.group_tables.clear()
        
        # 显示抽取信息
        if info_text:
            self.info_text.setText(info_text)
        
        # 显示备选专家名单
        if candidates and fields:
            self.candidates_table.setRowCount(len(candidates))
            self.candidates_table.setColumnCount(len(fields))
            self.candidates_table.setHorizontalHeaderLabels(fields)
            
            for row_idx, expert in enumerate(candidates):
                for col_idx, field in enumerate(fields):
                    value = expert.get(field, "")
                    item = QTableWidgetItem(str(value) if value is not None else "")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.candidates_table.setItem(row_idx, col_idx, item)
        
        # 显示分组结果
        if groups and fields:
            for group_name, judges in groups.items():
                table = QTableWidget()
                table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                table.setFont(QFont("SimHei", 10))
                table.setStyleSheet("""
                    QTableWidget {
                        border: 1px solid #cccccc;
                        border-radius: 4px;
                        background-color: #f9f9f9;
                        alternate-background-color: #f0f0f0;
                    }
                    QHeaderView::section {
                        background-color: #e0e0e0;
                        padding: 4px;
                        border: 1px solid #cccccc;
                    }
                """)
                table.setAlternatingRowColors(True)
                
                # 添加角色字段（如果不存在）
                display_fields = fields.copy()
                if "角色" not in display_fields:
                    display_fields.append("角色")
                
                table.setRowCount(len(judges))
                table.setColumnCount(len(display_fields))
                table.setHorizontalHeaderLabels(display_fields)
                
                for row_idx, judge in enumerate(judges):
                    for col_idx, field in enumerate(display_fields):
                        value = judge.get(field, "")
                        item = QTableWidgetItem(str(value) if value is not None else "")
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        table.setItem(row_idx, col_idx, item)
                
                self.results_tabs.addTab(table, group_name)
                self.group_tables[group_name] = table
        
        # 启用按钮
        self.avoidance_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.new_selection_btn.setEnabled(True)
    
    def on_avoidance_clicked(self):
        """打开规避设置弹窗"""
        if not self.fields:
            QMessageBox.warning(self, "警告", "没有可用的字段信息")
            return
            
        dialog = AvoidanceDialog(self.fields, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_avoidance_settings()
            if settings["values"]:  # 确保有选择规避值
                self.perform_avoidance.emit(settings)
    
    def get_selection_info(self):
        """获取抽取信息文本"""
        return self.info_text.toPlainText()