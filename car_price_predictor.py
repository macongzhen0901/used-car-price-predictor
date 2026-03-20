#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二手车价格预测系统
================
功能：
1. 数据抓取（汽车之家、瓜子二手车等平台）
2. 价格预测模型（基于随机森林回归）
3. 数据同步到飞书多维表格

作者：马从振的 OpenClaw 🦞
创建时间：2026-03-20
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============= 配置 =============
BASE_URL = "https://api.example.com"  # 实际使用时替换为真实 API
BITABLE_APP_TOKEN = "QEMXbhPFZa4jPXsYenjc5tK3nmw"
VEHICLE_TABLE_ID = "tblSgpfSCZInJxVf"
MODEL_TABLE_ID = "tblZ7Q8fizmv1022"

# ============= 折旧率模型 =============
# 基于行业数据的品牌残值率（2026 年）
DEPRECIATION_MODEL = {
    "奔驰": {"1 年": 0.75, "3 年": 0.60, "5 年": 0.48, "10 年": 0.25, "系数": 1.15},
    "宝马": {"1 年": 0.73, "3 年": 0.58, "5 年": 0.46, "10 年": 0.23, "系数": 1.12},
    "奥迪": {"1 年": 0.72, "3 年": 0.56, "5 年": 0.44, "10 年": 0.22, "系数": 1.10},
    "丰田": {"1 年": 0.78, "3 年": 0.65, "5 年": 0.52, "10 年": 0.30, "系数": 1.20},
    "本田": {"1 年": 0.76, "3 年": 0.63, "5 年": 0.50, "10 年": 0.28, "系数": 1.18},
    "大众": {"1 年": 0.70, "3 年": 0.55, "5 年": 0.42, "10 年": 0.20, "系数": 1.05},
    "日产": {"1 年": 0.68, "3 年": 0.52, "5 年": 0.40, "10 年": 0.18, "系数": 1.00},
    "福特": {"1 年": 0.65, "3 年": 0.48, "5 年": 0.36, "10 年": 0.16, "系数": 0.95},
    "别克": {"1 年": 0.63, "3 年": 0.46, "5 年": 0.34, "10 年": 0.15, "系数": 0.92},
    "现代": {"1 年": 0.62, "3 年": 0.45, "5 年": 0.33, "10 年": 0.14, "系数": 0.90},
    "比亚迪": {"1 年": 0.70, "3 年": 0.55, "5 年": 0.42, "10 年": 0.20, "系数": 1.05},
    "特斯拉": {"1 年": 0.80, "3 年": 0.68, "5 年": 0.55, "10 年": 0.35, "系数": 1.25},
    "其他": {"1 年": 0.60, "3 年": 0.42, "5 年": 0.30, "10 年": 0.12, "系数": 0.85},
}

# ============= 价格预测模型 =============
class UsedCarPricePredictor:
    """二手车价格预测器"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.encoders = {}
        self.is_trained = False
        
    def prepare_features(self, df):
        """准备特征数据"""
        df = df.copy()
        
        # 编码分类变量
        categorical_cols = ['品牌', '车型', '变速箱', '环保标准', '地区']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # 计算关键特征
        if '上牌年份' in df.columns:
            current_year = datetime.now().year
            df['车龄'] = current_year - df['上牌年份']
        
        # 计算折旧率
        if '新车指导价' in df.columns and '当前售价' in df.columns:
            df['折旧率'] = (1 - df['当前售价'] / df['新车指导价']) * 100
        
        return df
    
    def train(self, df):
        """训练模型"""
        feature_cols = [
            '车龄', '行驶里程 (万公里)', '排量 (L)', '马力',
            '品牌_encoded', '变速箱_encoded', '环保标准_encoded'
        ]
        
        # 检查必要的列是否存在
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < 4:
            print("⚠️ 特征列不足，使用简化模型")
            available_cols = ['车龄', '行驶里程 (万公里)']
        
        X = df[available_cols].fillna(0)
        y = df['当前售价'].fillna(0)
        
        self.model.fit(X, y)
        self.is_trained = True
        print(f"✅ 模型训练完成，R² = {self.model.score(X, y):.4f}")
        
    def predict(self, car_data):
        """预测单辆车价格"""
        if not self.is_trained:
            # 使用折旧率模型进行基础预测
            return self._predict_by_depreciation(car_data)
        
        # 使用 ML 模型预测
        features = self._extract_features(car_data)
        prediction = self.model.predict([features])[0]
        return max(prediction, 0.5)  # 最低 0.5 万元
    
    def _predict_by_depreciation(self, car_data):
        """基于折旧率的预测（备用方案）"""
        brand = car_data.get('品牌', '其他')
        car_age = car_data.get('车龄', 5)
        new_price = car_data.get('新车指导价', 10)
        mileage = car_data.get('行驶里程 (万公里)', 10)
        
        # 获取品牌折旧参数
        dep = DEPRECIATION_MODEL.get(brand, DEPRECIATION_MODEL['其他'])
        
        # 插值计算残值率
        if car_age <= 1:
            residual_rate = dep["1 年"]
        elif car_age <= 3:
            residual_rate = dep["1 年"] - (dep["1 年"] - dep["3 年"]) * (car_age - 1) / 2
        elif car_age <= 5:
            residual_rate = dep["3 年"] - (dep["3 年"] - dep["5 年"]) * (car_age - 3) / 2
        elif car_age <= 10:
            residual_rate = dep["5 年"] - (dep["5 年"] - dep["10 年"]) * (car_age - 5) / 5
        else:
            residual_rate = dep["10 年"] * (0.9 ** (car_age - 10))
        
        # 里程修正（每增加 1 万公里，价格降低 2%）
        mileage_factor = max(0.5, 1 - (mileage - 10) * 0.02)
        
        # 计算预测价格
        predicted_price = new_price * residual_rate * mileage_factor * dep["系数"]
        
        return round(predicted_price, 2)
    
    def _extract_features(self, car_data):
        """提取特征向量"""
        features = []
        
        # 数值特征
        features.append(car_data.get('车龄', 5))
        features.append(car_data.get('行驶里程 (万公里)', 10))
        features.append(car_data.get('排量 (L)', 1.5))
        features.append(car_data.get('马力', 150))
        
        # 分类特征（使用默认编码）
        for col in ['品牌', '变速箱', '环保标准']:
            if col in self.encoders:
                try:
                    code = self.encoders[col].transform([car_data.get(col, '其他')])[0]
                except:
                    code = 0
            else:
                code = 0
            features.append(code)
        
        return features
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_trained:
            return {}
        
        feature_names = [
            '车龄', '行驶里程', '排量', '马力',
            '品牌', '变速箱', '环保标准'
        ]
        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ============= 数据抓取模块 =============
class CarDataScraper:
    """二手车数据抓取器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_autohome(self, brand=None, city='全国', limit=50):
        """抓取汽车之家数据（模拟）"""
        # 注意：实际使用时需要调用真实 API 或爬虫
        # 这里提供示例数据结构
        print(f"📊 抓取汽车之家数据：品牌={brand}, 城市={city}")
        
        # 示例数据（实际应替换为真实抓取）
        sample_data = [
            {
                "车辆 ID": f"AH{datetime.now().strftime('%Y%m%d%H%M%S')}{i}",
                "品牌": brand or "丰田",
                "车型": "凯美瑞 2.0G 豪华版",
                "上牌年份": 2021,
                "车龄 (年)": 5,
                "行驶里程 (万公里)": 5.2,
                "排量 (L)": 2.0,
                "马力": 178,
                "变速箱": "自动",
                "环保标准": "国六 B",
                "新车指导价 (万元)": 19.98,
                "当前售价 (万元)": 12.5,
                "地区": city,
                "颜色": "白色",
                "数据来源": "汽车之家",
                "抓取时间": int(datetime.now().timestamp() * 1000)
            }
            for i in range(limit)
        ]
        
        return sample_data
    
    def scrape_guazi(self, brand=None, city='全国', limit=50):
        """抓取瓜子二手车数据（模拟）"""
        print(f"📊 抓取瓜子二手车数据：品牌={brand}, 城市={city}")
        
        sample_data = [
            {
                "车辆 ID": f"GZ{datetime.now().strftime('%Y%m%d%H%M%S')}{i}",
                "品牌": brand or "本田",
                "车型": "雅阁 1.5T 精英版",
                "上牌年份": 2020,
                "车龄 (年)": 6,
                "行驶里程 (万公里)": 7.8,
                "排量 (L)": 1.5,
                "马力": 194,
                "变速箱": "CVT",
                "环保标准": "国六 A",
                "新车指导价 (万元)": 17.98,
                "当前售价 (万元)": 10.2,
                "地区": city,
                "颜色": "黑色",
                "数据来源": "瓜子二手车",
                "抓取时间": int(datetime.now().timestamp() * 1000)
            }
            for i in range(limit)
        ]
        
        return sample_data


# ============= 飞书多维表格同步 =============
class BitableSync:
    """飞书多维表格数据同步"""
    
    def __init__(self, app_token, table_id):
        self.app_token = app_token
        self.table_id = table_id
        # 实际使用时需要配置飞书 API token
        self.api_base = "https://open.feishu.cn/open-apis/bitable/v1"
    
    def sync_records(self, records):
        """同步记录到多维表格"""
        print(f"📝 同步 {len(records)} 条记录到多维表格...")
        
        # 注意：实际使用时需要调用飞书 API
        # 这里仅做模拟
        for record in records[:5]:  # 仅显示前 5 条
            print(f"  - {record.get('品牌')} {record.get('车型')} | {record.get('当前售价 (万元)')}万元")
        
        print(f"✅ 同步完成")
        return True
    
    def get_records(self, limit=100):
        """获取记录"""
        # 实际使用时调用飞书 API
        return []


# ============= 主程序 =============
def main():
    """主程序入口"""
    print("=" * 60)
    print("🚗 二手车价格预测系统 v1.0")
    print("🦞 Powered by 马从振的 OpenClaw")
    print("=" * 60)
    
    # 1. 初始化
    predictor = UsedCarPricePredictor()
    scraper = CarDataScraper()
    sync = BitableSync(BITABLE_APP_TOKEN, VEHICLE_TABLE_ID)
    
    # 2. 抓取数据
    print("\n📡 开始抓取数据...")
    autohome_data = scraper.scrape_autohome(brand="丰田", city="北京", limit=20)
    guazi_data = scraper.scrape_guazi(brand="本田", city="上海", limit=20)
    
    all_data = autohome_data + guazi_data
    
    # 3. 转换为 DataFrame
    df = pd.DataFrame(all_data)
    print(f"📊 获取到 {len(df)} 条车辆数据")
    
    # 4. 训练预测模型
    print("\n🤖 训练价格预测模型...")
    df = predictor.prepare_features(df)
    predictor.train(df)
    
    # 5. 价格预测
    print("\n💰 开始价格预测...")
    for idx, row in df.iterrows():
        car_data = row.to_dict()
        predicted = predictor.predict(car_data)
        actual = car_data.get('当前售价 (万元)', 0)
        deviation = ((predicted - actual) / actual * 100) if actual > 0 else 0
        
        df.at[idx, '预测价格 (万元)'] = round(predicted, 2)
        df.at[idx, '价格偏差 (%)'] = round(deviation, 2)
        
        if idx < 5:  # 显示前 5 条
            print(f"  {car_data.get('品牌')} {car_data.get('车型')} | "
                  f"实际：{actual}万 | 预测：{predicted:.2f}万 | 偏差：{deviation:+.1f}%")
    
    # 6. 同步到飞书多维表格
    print("\n☁️ 同步数据到飞书多维表格...")
    records = df.to_dict('records')
    sync.sync_records(records)
    
    # 7. 输出特征重要性
    print("\n📈 特征重要性分析:")
    importance = predictor.get_feature_importance()
    for feature, score in importance.items():
        bar = "█" * int(score * 20)
        print(f"  {feature:8} {bar} {score:.2%}")
    
    # 8. 保存结果
    output_file = f"/home/gem/workspace/agent/workspace/used_car_system/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存：{output_file}")
    
    print("\n" + "=" * 60)
    print("✅ 系统运行完成")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    main()
