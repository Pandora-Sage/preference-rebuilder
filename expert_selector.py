import random
import pandas as pd
from datetime import datetime

class ExpertSelector:
    """专家抽取核心逻辑处理器"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.selected_technical = []  # 选中的技术评委
        self.selected_venture = []    # 选中的创投评委
        self.technical_candidates = []  # 技术备选评委
        self.venture_candidates = []    # 创投备选评委
        self.groups = [               # 7个固定比赛场地
            "企业1组", "企业2组", "企业3组", "企业4组",
            "团队1组", "团队2组", "团队3组"
        ]
        self.substitute_records = []  # 替补记录
        self.original_params = None   # 原始抽取参数
        self.selection_results = None # 最终抽取结果
    
    def select_experts(self, params):
        """执行专家抽取（支持人数不足时从备选补抽）"""
        # 重置之前的抽取结果
        self.technical_candidates = []
        self.venture_candidates = []
        self.selected_technical = []
        self.selected_venture = []
        self.groups = {}
        self.substitute_records = []
        self.original_params = params  # 保存原始参数
        
        if self.data_manager.df is None or self.data_manager.df.empty:
            raise Exception("专家库中没有数据，请先加载专家库")
        
        # 获取筛选条件
        directions = params.get('professional_directions', [])
        avoid_units = params.get('avoid_units', [])
        technical_ratio = params.get('technical_ratio', 5)
        venture_ratio = params.get('venture_capital_ratio', 5)
        
        # 1. 应用专业方向和规避单位筛选
        filtered_df = self.data_manager.df.copy()
        if '专业方向' in filtered_df.columns and directions:
            filtered_df = filtered_df[filtered_df['专业方向'].isin(directions)]
        if '单位' in filtered_df.columns and avoid_units:
            filtered_df = filtered_df[~filtered_df['单位'].isin(avoid_units)]
        
        if filtered_df.empty:
            raise Exception("筛选后没有符合条件的专家，请调整筛选条件")
        
        # 2. 基础参数定义（技术21人，创投14人，共35人）
        needed_technical = 21
        needed_venture = 14
        total_needed = needed_technical + needed_venture
        filtered_count = len(filtered_df)
        
        # 3. 生成备选名单（确保备选池足够大，优先包含所有筛选后专家）
        # 计算所需备选总数（按比例）
        total_candidate_ratio = (needed_technical * technical_ratio) + (needed_venture * venture_ratio)
        # 备选名单至少包含所有筛选后专家（确保有足够补抽资源）
        total_candidates = filtered_df.sample(n=min(total_candidate_ratio, filtered_count), replace=False).to_dict('records')
        
        # 4. 第一次抽取：优先从筛选后专家中抽取
        first_draw_count = min(filtered_count, total_needed)
        first_draw = random.sample(total_candidates, first_draw_count)  # 从备选池中抽初始人选
        
        # 5. 检查是否需要补抽
        if first_draw_count < total_needed:
            # 计算差额
            lack_count = total_needed - first_draw_count
            self.substitute_records.append(f"第一次抽取仅获得{first_draw_count}人，需补抽{lack_count}人")
            
            # 从备选名单中补抽（排除已选中的）
            remaining_candidates = [exp for exp in total_candidates if exp not in first_draw]
            if len(remaining_candidates) < lack_count:
                # 若备选仍不足，尽可能补抽
                lack_count = len(remaining_candidates)
                self.substitute_records.append(f"备选名单不足，仅能补抽{lack_count}人")
            
            # 执行补抽
            supplement_draw = random.sample(remaining_candidates, lack_count) if lack_count > 0 else []
            selected_all = first_draw + supplement_draw
        else:
            selected_all = first_draw
        
        # 6. 分配技术和创投评委（按比例分配实际选中人数）
        actual_total = len(selected_all)
        # 按21:14的比例分配（即3:2）
        technical_ratio_actual = needed_technical / total_needed
        technical_count = min(needed_technical, int(actual_total * technical_ratio_actual))
        venture_count = actual_total - technical_count
        
        # 确保技术和创投人数不超过理论最大值
        technical_count = min(technical_count, needed_technical)
        venture_count = min(venture_count, needed_venture)
        
        # 随机分配角色
        random.shuffle(selected_all)
        self.selected_technical = selected_all[:technical_count]
        self.selected_venture = selected_all[technical_count:technical_count + venture_count]
        
        # 7. 更新备选名单（排除已选中的）
        self.technical_candidates = [exp for exp in total_candidates if exp not in self.selected_technical]
        self.venture_candidates = [exp for exp in total_candidates if exp not in self.selected_venture]
        
        # 8. 分组（处理人数不足时的分组逻辑）
        self._group_experts()
        
        # 9. 记录实际抽取结果
        self.substitute_records.append(
            f"最终抽取结果：技术评委{len(self.selected_technical)}人，创投评委{len(self.selected_venture)}人，共{actual_total}人"
        )
        
        return selected_all
        
    def _generate_candidates(self, pool, selected, required, ratio):
        """生成备选名单（排除已选中专家）"""
        candidates = [exp for exp in pool if exp not in selected]
        needed = required * ratio
        if len(candidates) < needed:
            raise Warning(f"备选专家不足，需{needed}名，仅能提供{len(candidates)}名")
        return random.sample(candidates, min(needed, len(candidates)))
    
    def _group_experts(self):
        """将选中的专家分组（兼容人数不足的情况）"""
        groups = [
            "企业1组", "企业2组", "企业3组", "企业4组",
            "团队1组", "团队2组", "团队3组"
        ]
        group_count = len(groups)
        
        # 计算每组实际可分配的技术/创投评委数量（向下取整，剩余的均匀分配）
        tech_per_group = len(self.selected_technical) // group_count
        tech_remaining = len(self.selected_technical) % group_count
        
        venture_per_group = len(self.selected_venture) // group_count
        venture_remaining = len(self.selected_venture) % group_count
        
        # 复制列表用于分配（避免修改原列表）
        technical_available = self.selected_technical.copy()
        venture_available = self.selected_venture.copy()
        
        for i, group in enumerate(groups):
            # 分配技术评委（前tech_remaining组多1人）
            tech_in_group = tech_per_group + (1 if i < tech_remaining else 0)
            tech_selected = random.sample(technical_available, tech_in_group) if tech_in_group > 0 else []
            for exp in tech_selected:
                technical_available.remove(exp)
                exp['角色'] = '技术评委'
            
            # 分配创投评委（前venture_remaining组多1人）
            venture_in_group = venture_per_group + (1 if i < venture_remaining else 0)
            venture_selected = random.sample(venture_available, venture_in_group) if venture_in_group > 0 else []
            for exp in venture_selected:
                venture_available.remove(exp)
                exp['角色'] = '创投评委'
            
            self.groups[group] = tech_selected + venture_selected
    
    def perform_secondary_avoidance(self, settings):
        """执行二次回避（根据字段值筛选）"""
        if self.selection_results is None:
            return False, "无抽取结果可执行回避"
        
        field = settings.get("field")
        avoid_values = settings.get("values", [])
        if not field or not avoid_values:
            return False, "请设置有效的规避字段和值"
        
        # 筛选需要回避的专家
        avoid_technical = [
            exp for exp in self.selected_technical
            if str(exp.get(field, "")).strip() in avoid_values
        ]
        avoid_venture = [
            exp for exp in self.selected_venture
            if str(exp.get(field, "")).strip() in avoid_values
        ]
        
        if not avoid_technical and not avoid_venture:
            return False, "未找到需要回避的专家"
        
        # 执行替补
        self._substitute_experts(
            avoid_technical, self.selected_technical, 
            self.technical_candidates, 21, "技术"
        )
        self._substitute_experts(
            avoid_venture, self.selected_venture, 
            self.venture_candidates, 14, "创投"
        )
        
        # 更新分组结果
        grouped_results = self._group_experts()
        self.selection_results = pd.DataFrame(grouped_results)
        return True, f"成功回避{len(avoid_technical)+len(avoid_venture)}名专家，已完成替补"
    
    def _substitute_experts(self, avoid_list, selected_list, candidate_pool, required_total, expert_type):
        """执行替补逻辑（从对应类型备选池抽取）"""
        need_replace = len(avoid_list)
        if need_replace == 0:
            return
        
        self.substitute_records.append(
            f"{expert_type}评委需回避{need_replace}人：{[exp.get('姓名', '未知') for exp in avoid_list]}"
        )
        
        # 移除需回避的专家
        remaining = [exp for exp in selected_list if exp not in avoid_list]
        
        # 筛选可用替补（未被选中且不在回避列表）
        available_substitutes = [
            exp for exp in candidate_pool 
            if exp not in selected_list and exp not in avoid_list
        ]
        
        if len(available_substitutes) < need_replace:
            raise Warning(f"{expert_type}评委替补不足，需{need_replace}名，仅{len(available_substitutes)}名可用")
        
        # 随机抽取替补
        substitutes = random.sample(available_substitutes, need_replace)
        self.substitute_records.append(
            f"替补{expert_type}评委{len(substitutes)}人：{[exp.get('姓名', '未知') for exp in substitutes]}"
        )
        
        # 更新选中列表（确保总数正确）
        updated = remaining + substitutes
        if len(updated) > required_total:
            updated = updated[:required_total]
        
        if expert_type == "技术":
            self.selected_technical = updated
        else:
            self.selected_venture = updated
    
    def generate_selection_info(self, params, timestamp):
        """生成抽取信息文本"""
        info = [
            f"抽取时间：{timestamp}",
            f"抽取事由：{params.get('selection_reason', '未说明')}",
            f"技术评委：{len(self.selected_technical)}名（7组×3名）",
            f"创投评委：{len(self.selected_venture)}名（7组×2名）",
            f"技术备选比例：1:{params.get('technical_ratio', 5)}，共{len(self.technical_candidates)}名",
            f"创投备选比例：1:{params.get('venture_capital_ratio', 5)}，共{len(self.venture_candidates)}名"
        ]
        if self.substitute_records:
            info.append("\n替补记录：")
            info.extend(self.substitute_records)
        return "\n".join(info)
    
    def get_groups(self):
        """获取分组列表"""
        return self.groups
    
    def get_selection_results(self):
        """获取抽取结果"""
        return self.selection_results
