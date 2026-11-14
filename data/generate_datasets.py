# -*- coding: utf-8 -*-
"""
DKNet 高质量数据集生成脚本
使用模板化生成策略构建 D1（微博时序）和 D2（新闻门户）数据集

核心策略：
  - 模板化生成（非Faker随机文本），为10个话题各准备模板
  - 注入真实实体（人名、地名、机构名），模拟微博和新闻风格
  - 话题生命周期模拟 + 概念漂移
"""

import os
import sys
import json
import csv
import hashlib
import argparse
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

# ============================================================
# 全局配置
# ============================================================

# 10个话题定义
TOPIC_DEFINITIONS = {
    "科技": {
        "keywords": [
            "人工智能", "机器学习", "深度学习", "大模型", "ChatGPT",
            "自动驾驶", "芯片", "半导体", "量子计算", "5G",
            "云计算", "大数据", "区块链", "物联网", "机器人",
            "虚拟现实", "增强现实", "元宇宙", "数字化转型", "智能终端",
        ],
        "entities": {
            "person": ["黄仁勋", "马斯克", "任正非", "雷军", "李彦宏",
                       "张一鸣", "王兴", "马化腾", "扎克伯格", "库克",
                       "奥特曼", "李飞飞", "周鸿祎", "刘强东", "沈南鹏"],
            "org": ["华为", "阿里巴巴", "腾讯", "百度", "字节跳动",
                    "小米", "OpenAI", "英伟达", "台积电", "中芯国际",
                    "大疆创新", "商汤科技", "科大讯飞", "寒武纪", "地平线"],
            "location": ["深圳", "北京", "上海", "硅谷", "中关村",
                         "杭州", "成都", "武汉", "合肥", "台北"],
        },
        "weibo_templates": [
            "刚刚看到{entity}在{location}发布的新产品，真的绝了！{keyword}技术又上了一个台阶🔥 #科技创新#",
            "{keyword}领域又出大新闻了！{entity}宣布最新突破，大家怎么看？ https://t.cn/A6xKm9pQ",
            "今天参加了{entity}的发布会，{keyword}的演示太震撼了👍 未来已来！ #科技改变生活#",
            "@科技日报 {entity}刚刚发布{keyword}白皮书，业内都在讨论这个方向💪",
            "谁说{keyword}离我们很远？{entity}已经在{location}落地应用了，亲测好用😂",
            "被{entity}的{keyword}技术圈粉了！这才是真正的硬核科技 #中国芯#",
            "听说{entity}要在{location}建新研发中心，专注{keyword}，期待！ https://t.cn/Bk3mN7p",
            "{keyword}赛道越来越卷了，{entity}这波操作直接封神🔥🔥🔥",
            "在{location}体验了{entity}的{keyword}产品，只能说一句：太厉害了！ #黑科技#",
            "@央视新闻 报道了{entity}在{keyword}领域的重大突破，民族自豪感拉满💪💪",
            "刚看完{entity}关于{keyword}的演讲PPT，信息量太大了，需要消化一下...",
            "朋友们，{keyword}时代真的来了！{entity}最新数据说明一切 #未来科技#",
            "笑死，还有人觉得{keyword}是噱头？看看{entity}的成绩单吧😂😂",
            "{entity}的{entity}在{location}展示了{keyword}最新成果，现场掌声雷动👏",
            "重磅！{entity}完成新一轮融资，全力押注{keyword}方向 https://t.cn/Mk9pQ2x",
            "今天{entity}开源了{keyword}框架，开发者社区炸了🔥 感谢开源精神！",
            "实测{entity}的{keyword}产品，速度快了3倍，价格降了一半，真香！",
            "{keyword}+{keyword}=? {entity}给出了答案，在{location}正式发布 #跨界融合#",
            "@人民日报 点赞{entity}在{keyword}领域的自主创新，这才是企业担当👍",
            "从{location}传来的消息：{entity}{keyword}项目通过验收，又一个里程碑！",
        ],
        "news_templates": [
            "{entity}发布{keyword}年度报告：行业规模突破万亿大关",
            "据新华社报道，{entity}在{location}正式启动{keyword}国家重点实验室建设",
            "业内人士指出，{keyword}技术已进入产业化关键期，{entity}等企业引领发展",
            "据悉，{entity}与{location}市政府签署战略合作协议，共建{keyword}产业生态",
            "据相关统计，2023年{keyword}领域投融资总额同比增长47%，{entity}领跑行业",
            "{entity}创始人{entity}在{location}峰会上发表主旨演讲，畅谈{keyword}未来十年",
            "据科技部消息，{entity}牵头的{keyword}重大项目通过国家验收",
            "从{location}获悉，{entity}自主研发的{keyword}核心芯片实现量产突破",
            "据工信部发布的数据显示，{keyword}产业规模连续三年保持两位数增长",
            "据相关人士透露，{entity}计划在{location}投资百亿建设{keyword}研发基地",
            "据权威机构评估，{entity}在{keyword}领域的专利数量位居全球前列",
            "据{location}科技局消息，{entity}主导的{keyword}国际标准正式发布",
            "据报道，{entity}联合多家机构成立{keyword}创新联合体，加速技术攻关",
            "据相关统计，{entity}的{keyword}产品已覆盖全球150多个国家和地区",
            "据业内分析，{keyword}赛道将迎来新一轮洗牌，{entity}具备先发优势",
        ],
        "lifecycle": {"start_week": 1, "peak_week": 8, "end_week": 48},
    },
    "经济": {
        "keywords": [
            "股市", "A股", "投资", "金融", "银行",
            "货币", "通胀", "GDP", "经济增长", "贸易",
            "消费", "房地产", "利率", "汇率", "基金",
            "债券", "期货", "独角兽", "IPO", "供应链",
        ],
        "entities": {
            "person": ["易纲", "郭树清", "刘鹤", "黄奇帆", "林毅夫",
                       "任泽平", "但斌", "张磊", "沈南鹏", "巴菲特",
                       "芒格", "达利欧", "鲍威尔", "拉加德", "田惠宇"],
            "org": ["中国人民银行", "证监会", "银保监会", "上交所", "深交所",
                    "工商银行", "建设银行", "高盛", "摩根大通", "中金公司",
                    "红杉资本", "黑石集团", "碧桂园", "万科", "比亚迪"],
            "location": ["上海", "深圳", "北京", "香港", "华尔街",
                         "陆家嘴", "金融街", "广州", "新加坡", "伦敦"],
        },
        "weibo_templates": [
            "今天A股又涨了！{keyword}板块领涨，{entity}大涨5%🔥 大家都回血了吗？",
            "刚看完{entity}关于{keyword}的分析报告，观点很犀利，分享给大家 https://t.cn/K9mP3xQ",
            "@财经网 {entity}最新发声：{keyword}趋势不可逆转，长期看好中国经济💪",
            "{keyword}又上热搜了！{entity}这波操作直接封神，散户笑了😂😂",
            "在{location}参加了{entity}的投资者交流会，{keyword}前景被反复提及 #价值投资#",
            "朋友们注意了！{entity}发布{keyword}预警信号，注意风险控制 https://t.cn/Bn7kP2m",
            "谁懂啊，{keyword}跌了一个月终于反弹了，{entity}功不可没👍",
            "据{entity}预测，{keyword}将在下半年迎来拐点，你怎么看？ #经济形势#",
            "{entity}在{location}表示：{keyword}政策将持续发力，稳增长决心不变",
            "刚入手{entity}推荐的{keyword}基金，坐等收益😂 希望别打脸",
            "重磅！{entity}宣布{keyword}新规落地，行业格局将发生重大变化🔥",
            "@央视财经 {entity}解读{keyword}数据：经济复苏信号明显 #宏观经济#",
            "{keyword}市场太刺激了，{entity}的预测又准了，佩服佩服👏👏",
            "在{location}出差，明显感受到{keyword}带来的变化，经济活力十足",
            "笑死，那些说{keyword}要崩的人，看看{entity}的最新数据吧😂",
            "今天{entity}在{location}论坛上关于{keyword}的发言太精彩了，全程做笔记",
            "{keyword}+{keyword}双轮驱动，{entity}发布最新经济展望报告 https://t.cn/Mk9pQ3x",
            "实测{entity}的{keyword}策略，三个月收益跑赢大盘，真香！ #理财心得#",
            "@人民日报 经济日报头版：{entity}谈{keyword}——高质量发展之路",
            "从{location}传来好消息：{entity}{keyword}指数创年内新高，市场信心回升💪",
        ],
        "news_templates": [
            "据国家统计局数据，今年一季度GDP同比增长5.3%，{keyword}贡献显著",
            "据中国人民银行消息，{entity}决定调整{keyword}政策利率，释放稳增长信号",
            "据相关统计，{entity}发布的{keyword}指数连续三个月回升，经济回暖态势明显",
            "据悉，{entity}在{location}举行年度{keyword}大会，发布行业白皮书",
            "业内人士指出，{keyword}市场已进入调整期，{entity}等机构建议理性投资",
            "据新华社报道，{entity}在{location}签署{keyword}合作协议，深化区域经济协作",
            "据{entity}最新研究报告，{keyword}领域投资热度持续攀升，同比增长32%",
            "据财政部消息，{entity}出台{keyword}新政策，支持实体经济发展",
            "从{location}获悉，{entity}正式挂牌成立，专注{keyword}领域投资",
            "据相关人士透露，{entity}计划在{location}设立{keyword}创新试验区",
            "据权威机构评估，{entity}在{keyword}改革方面取得阶段性成果",
            "据{location}金融监管局消息，{entity}获批开展{keyword}业务试点",
            "据报道，{entity}联合多部门发布{keyword}五年规划，明确发展路线图",
            "据相关统计，{entity}的{keyword}业务规模已突破千亿大关",
            "据业内分析，{keyword}政策调整将对市场产生深远影响，{entity}率先响应",
        ],
        "lifecycle": {"start_week": 1, "peak_week": 12, "end_week": 50},
    },
    "社会": {
        "keywords": [
            "就业", "住房", "医疗", "教育公平", "养老",
            "社会保障", "人口", "城市化", "乡村振兴", "社区治理",
            "食品安全", "公共安全", "交通出行", "垃圾分类", "生育政策",
            "收入分配", "消费升级", "新市民", "数字鸿沟", "共同富裕",
        ],
        "entities": {
            "person": ["张维为", "金灿荣", "郑永年", "温铁军", "贺雪峰",
                       "李玲", "姚洋", "管清友", "迟福林", "蔡昉",
                       "胡锡进", "白岩松", "敬一丹", "撒贝宁", "倪萍"],
            "org": ["民政部", "人社部", "住建部", "卫健委", "教育部",
                    "全国总工会", "国务院发展研究中心", "中国社会科学院",
                    "各地民政局", "社区服务中心", "红十字会", "慈善总会"],
            "location": ["北京", "上海", "广州", "深圳", "成都",
                         "杭州", "武汉", "重庆", "西安", "郑州"],
        },
        "weibo_templates": [
            "今天在{location}看到{keyword}的新变化，真的越来越好了👍 #民生改善#",
            "@人民日报 {entity}谈{keyword}：让每个人都能享受到发展成果💪",
            "{keyword}话题又上热搜了！{entity}的观点引发热议，大家怎么看？",
            "刚看完关于{keyword}的纪录片，感触很深...希望更多人关注这个问题 https://t.cn/K9mP3xQ",
            "在{location}亲身体验了{keyword}改革，效率提升太明显了！👏👏",
            "笑死，{entity}说{keyword}问题需要时间解决，评论区直接炸了😂",
            "朋友们，{keyword}新政今天正式实施了，快看看对你有什么影响 #政策解读#",
            "据{entity}调研，{keyword}满意度达到历史新高，为进步点赞🔥",
            "{entity}在{location}调研{keyword}工作，强调要坚持以人民为中心",
            "今天参加了{entity}组织的{keyword}公益活动，收获满满！ #志愿服务#",
            "重磅！{entity}发布{keyword}年度报告，数据令人振奋 https://t.cn/Bn7kP2m",
            "谁懂啊，{keyword}终于有改善了！{entity}这波操作太给力了👍👍",
            "@央视新闻 {entity}解读{keyword}新规，这些变化与你息息相关",
            "{keyword}+{keyword}，{entity}在{location}的创新做法值得推广 #社会治理#",
            "在{location}生活了十年，见证了{keyword}的巨大变化，感慨万千...",
            "刚看到{entity}关于{keyword}的提案，说到了老百姓的心坎上💪💪",
            "实测{entity}的{keyword}便民服务，从跑三次到一次都不用跑，真香！",
            "今天{entity}在{location}宣布{keyword}新举措，惠及百万群众🔥",
            "@新华网 {entity}：{keyword}是最大的民生工程，必须持续用力",
            "从{location}传来消息：{entity}{keyword}项目获评全国示范，太棒了👏",
        ],
        "news_templates": [
            "据新华社报道，{entity}在{location}召开{keyword}工作推进会，部署重点任务",
            "据{entity}发布的数据显示，{keyword}覆盖率已达到95%以上，成效显著",
            "据悉，{entity}联合多部门出台{keyword}新政策，着力解决群众急难愁盼问题",
            "业内人士指出，{keyword}改革进入深水区，{entity}提出系统性解决方案",
            "据相关统计，{entity}在{location}实施的{keyword}试点项目惠及50万群众",
            "据{entity}调研报告，{keyword}领域仍存在区域发展不平衡问题",
            "从{location}获悉，{entity}创新{keyword}服务模式，获群众广泛好评",
            "据民政部消息，{entity}主导的{keyword}标准正式实施，填补行业空白",
            "据报道，{entity}在{location}启动{keyword}专项行动，为期三个月",
            "据相关人士透露，{entity}将加大对{keyword}领域的财政投入力度",
            "据权威机构评估，{entity}推动的{keyword}改革成效位居全国前列",
            "据{location}政府消息，{entity}获评{keyword}工作先进典型",
            "据报道，{entity}联合高校开展{keyword}研究，为政策制定提供科学依据",
            "据相关统计，{entity}的{keyword}服务已覆盖全国300多个城市",
            "据业内分析，{keyword}将成为未来五年社会建设的重点方向",
        ],
        "lifecycle": {"start_week": 2, "peak_week": 15, "end_week": 50},
    },
    "娱乐": {
        "keywords": [
            "电影", "电视剧", "综艺", "音乐", "明星",
            "演唱会", "票房", "热搜", "粉丝", "偶像",
            "选秀", "真人秀", "动漫", "游戏", "短视频",
            "直播", "网红", "影视", "导演", "演员",
        ],
        "entities": {
            "person": ["吴京", "沈腾", "黄渤", "贾玲", "张艺谋",
                       "周杰伦", "刘德华", "杨幂", "赵丽颖", "王一博",
                       "肖战", "迪丽热巴", "易烊千玺", "周冬雨", "汤唯"],
            "org": ["万达影业", "华谊兄弟", "光线传媒", "爱奇艺", "腾讯视频",
                    "优酷", "芒果TV", "网易云音乐", "抖音", "哔哩哔哩",
                    "北京电影学院", "中央音乐学院", "中国电影集团", "央视综艺"],
            "location": ["北京", "上海", "横店", "香港", "洛杉矶",
                         "戛纳", "威尼斯", "东京", "首尔", "台北"],
        },
        "weibo_templates": [
            "啊啊啊！{entity}的新电影太好看了！{keyword}直接封神🔥🔥🔥 #必看推荐#",
            "刚看完{entity}在{location}的演唱会，现场太炸了！{keyword}全场大合唱😭",
            "@人民日报 {entity}主演的电影票房破30亿，{keyword}题材再创纪录💪",
            "谁懂啊，{entity}的{keyword}舞台太绝了，反复看了十几遍😂😂",
            "今天在{location}偶遇{entity}了！真人比照片还好看！{keyword} #追星日常#",
            "笑死，{entity}在{keyword}里的表演太搞笑了，笑到肚子疼😂😂😂",
            "朋友们！{entity}官宣了！{keyword}即将上线，期待值拉满！ https://t.cn/K9mP3xQ",
            "据{entity}透露，{keyword}第二季正在筹备中，原班人马回归👍",
            "{entity}在{location}电影节斩获大奖，{keyword}获国际认可👏👏👏",
            "今天刷到{entity}的{keyword}短视频，创意太棒了，转发了！ #创意无限#",
            "重磅！{entity}加盟{keyword}，这阵容也太豪华了吧🔥🔥",
            "@央视新闻 {entity}做客{keyword}特别节目，分享心路历程",
            "{keyword}排行榜出炉！{entity}霸榜，实至名归💪💪",
            "在{location}看了{entity}的{keyword}首映礼，口碑炸裂，预测票房大爆",
            "刚看完{entity}的{keyword}综艺，笑到停不下来，强烈推荐！ #下饭综艺#",
            "实测{entity}推荐的{keyword}，真的上头，根本停不下来😂",
            "今天{entity}在{location}举办{keyword}见面会，粉丝从全国各地赶来🔥",
            "@新华网 {entity}：用{keyword}讲好中国故事，传播正能量",
            "从{location}传来消息：{entity}{keyword}作品入围国际大奖，为国争光👏",
        ],
        "news_templates": [
            "据国家电影局数据，{entity}主演的{keyword}影片票房突破40亿元",
            "据新华社报道，{entity}在{location}国际电影节荣获最佳影片奖",
            "据悉，{entity}宣布退出{keyword}节目录制，引发粉丝热议",
            "业内人士指出，{keyword}市场正经历结构性调整，{entity}积极转型",
            "据相关统计，{entity}的{keyword}作品全球播放量突破50亿次",
            "据{entity}官方消息，{keyword}巡演{location}站门票三秒售罄",
            "从{location}获悉，{entity}签约{keyword}新项目，投资规模达5亿元",
            "据广电总局消息，{entity}出品的{keyword}作品获年度优秀节目奖",
            "据报道，{entity}在{location}举办{keyword}发布会，宣布全新战略布局",
            "据相关人士透露，{entity}将担任{keyword}总导演，预计明年上映",
            "据权威机构评估，{entity}在{keyword}领域的影响力指数位居行业首位",
            "据{location}文化部门消息，{entity}获评{keyword}文化推广大使",
            "据报道，{entity}联合{keyword}平台推出原创内容扶持计划",
            "据相关统计，{entity}的{keyword}社交媒体粉丝数突破一亿",
            "据业内分析，{keyword}行业将迎来新一轮整合，{entity}优势明显",
        ],
        "lifecycle": {"start_week": 3, "peak_week": 10, "end_week": 48},
    },
    "政治": {
        "keywords": [
            "政策", "法律", "政府", "改革", "治理",
            "法治", "廉政", "选举", "外交", "国际关系",
            "国家安全", "统一", "制度", "民主", "监督",
            "协商", "基层", "党建", "乡村振兴", "高质量发展",
        ],
        "entities": {
            "person": ["王毅", "秦刚", "华春莹", "汪文斌", "耿爽",
                       "崔天凯", "乐玉成", "杨洁篪", "王沪宁", "韩正"],
            "org": ["国务院", "全国人大", "全国政协", "中央纪委", "外交部",
                    "国防部", "最高人民法院", "最高人民检察院",
                    "国家发改委", "国家监委", "中央党校", "各省委省政府"],
            "location": ["北京", "上海", "广州", "深圳", "雄安新区",
                         "海南自贸港", "粤港澳大湾区", "长三角", "成渝", "西部"],
        },
        "weibo_templates": [
            "@人民日报 {entity}就{keyword}发表重要谈话，释放积极信号💪 #大国担当#",
            "今天{entity}在{location}考察{keyword}工作，为基层干部点赞👍",
            "重磅！{entity}通过{keyword}新法案，这些变化与你息息相关 https://t.cn/K9mP3xQ",
            "据{entity}消息，{keyword}改革取得重大进展，为发展注入新动力🔥",
            "在{location}见证了{keyword}的落地实施，效率真的很高👏👏",
            "朋友们注意了！{entity}发布{keyword}新规，2024年起正式施行 #政策速递#",
            "笑死，外媒又在炒作{keyword}话题，{entity}直接回应了😂",
            "{entity}在{location}强调：{keyword}是国家的根本，必须坚定不移",
            "今天{entity}出席{keyword}大会并作重要讲话，全场掌声雷动💪💪",
            "实测{entity}推出的{keyword}便民措施，办事效率提升了好几个档次！",
            "@央视新闻 {entity}会见外宾，就{keyword}达成重要共识 #外交风云#",
            "{keyword}+{keyword}，{entity}在{location}的创新实践获全国推广 #改革先锋#",
            "从{location}传来消息：{entity}{keyword}工作获中央肯定，太棒了👏",
            "今天{entity}在{location}宣布{keyword}新举措，惠及亿万群众🔥",
            "@新华社 {entity}谈{keyword}：以人民为中心，让发展成果惠及全体人民",
            "刚看完{entity}关于{keyword}的专访，格局太大了，受益匪浅...",
        ],
        "news_templates": [
            "据新华社报道，{entity}在{location}主持召开{keyword}工作会议，部署重点任务",
            "据{entity}消息，{keyword}条例经审议通过，将于下月起正式施行",
            "据悉，{entity}在{location}调研{keyword}工作，强调要深化改革创新",
            "业内人士指出，{keyword}改革已进入攻坚期，{entity}提出新思路",
            "据相关统计，{entity}推动的{keyword}改革惠及全国3亿群众",
            "据全国人大消息，{entity}提交的{keyword}议案获高票通过",
            "从{location}获悉，{entity}创新{keyword}治理模式，成为全国标杆",
            "据{entity}发布的数据显示，{keyword}领域群众满意度达98.5%",
            "据报道，{entity}在{location}启动{keyword}综合改革试点",
            "据相关人士透露，{entity}将出台{keyword}配套政策，进一步完善制度体系",
            "据权威机构评估，{entity}在{keyword}领域的改革成效位居全国前列",
            "据{location}政府消息，{entity}获评{keyword}工作先进典型",
            "据报道，{entity}联合多部门开展{keyword}专项督查，确保政策落地",
            "据相关统计，{entity}推动的{keyword}创新做法已在全国200多个城市推广",
            "据业内分析，{keyword}将成为未来五年全面深化改革的重点领域",
        ],
        "lifecycle": {"start_week": 1, "peak_week": 20, "end_week": 52},
    },
    "体育": {
        "keywords": [
            "足球", "篮球", "乒乓球", "羽毛球", "游泳",
            "田径", "奥运会", "世界杯", "亚运会", "NBA",
            "中超", "CBA", "马拉松", "电竞", "冬奥",
            "全民健身", "体育产业", "运动员", "教练", "冠军",
        ],
        "entities": {
            "person": ["梅西", "C罗", "姆巴佩", "库里", "詹姆斯",
                       "姚明", "刘翔", "苏炳添", "全红婵", "谷爱凌",
                       "郑钦文", "潘展乐", "马龙", "樊振东", "张雨霏"],
            "org": ["国际奥委会", "FIFA", "NBA", "中国足协", "国家体育总局",
                    "中国奥委会", "中超联赛", "CBA联盟", "WTA", "ATP"],
            "location": ["北京", "上海", "广州", "深圳", "巴黎",
                         "洛杉矶", "东京", "多哈", "杭州", "成都"],
        },
        "weibo_templates": [
            "啊啊啊！{entity}又赢了！{keyword}太精彩了🔥🔥🔥 #冠军之路#",
            "刚看完{entity}在{location}的比赛，{keyword}直接封神！太激动了😭😭",
            "@央视体育 {entity}夺得{keyword}金牌，为国争光💪💪💪",
            "谁懂啊，{entity}的{keyword}操作太秀了，反复看了几十遍😂😂",
            "今天在{location}现场看{entity}比赛，氛围太燃了！{keyword} #现场直击#",
            "笑死，{entity}在{keyword}赛后的采访太搞笑了，性格太好了😂😂😂",
            "朋友们！{entity}官宣加盟新球队！{keyword}新赛季值得期待！ https://t.cn/K9mP3xQ",
            "据{entity}透露，{keyword}集训进展顺利，状态正佳👍",
            "{entity}在{location}打破{keyword}世界纪录，历史性一刻👏👏👏",
            "今天刷到{entity}的{keyword}训练视频，太拼了，致敬！ #体育精神#",
            "重磅！{entity}宣布退役，{keyword}生涯完美谢幕🔥 #传奇落幕#",
            "@人民日报 {entity}谈{keyword}：体育强则中国强",
            "{keyword}赛程出炉！{entity}首战对手强劲，你预测谁赢？ #赛事预测#",
            "在{location}看了{entity}的{keyword}决赛，现场三万人齐声呐喊🔥",
            "刚看完{entity}的{keyword}纪录片，泪目了，这才是真正的冠军精神😭",
            "实测{entity}推荐的{keyword}训练方法，坚持一个月效果明显👍",
            "今天{entity}在{location}举办{keyword}训练营，小球员们太开心了",
            "@新华社 {entity}：用{keyword}激励青少年，让体育精神薪火相传",
            "从{location}传来消息：{entity}{keyword}成绩创个人最佳，继续加油💪",
        ],
        "news_templates": [
            "据新华社报道，{entity}在{location}{keyword}比赛中夺得金牌，实现历史性突破",
            "据{entity}消息，{keyword}赛事筹备工作全面就绪，届时将有150个国家参赛",
            "据悉，{entity}正式获得{location}{keyword}赛事举办权",
            "业内人士指出，{keyword}产业规模持续扩大，{entity}预计未来五年翻一番",
            "据相关统计，{entity}在{keyword}项目上的投入同比增长35%",
            "据国家体育总局消息，{entity}入选{keyword}国家队集训名单",
            "从{location}获悉，{entity}以破纪录成绩夺得{keyword}冠军",
            "据{entity}官方消息，{keyword}新赛季赛程正式公布",
            "据报道，{entity}在{location}召开{keyword}发展大会，发布五年规划",
            "据相关人士透露，{entity}将申办{keyword}国际赛事",
            "据权威机构评估，{entity}在{keyword}领域的综合实力位居亚洲前列",
            "据{location}体育局消息，{entity}获评{keyword}运动推广先进城市",
            "据报道，{entity}联合{keyword}平台推出青少年体育发展计划",
            "据相关统计，{entity}的{keyword}赛事观众人数突破千万",
            "据业内分析，{keyword}将成为体育产业新的增长极",
        ],
        "lifecycle": {"start_week": 5, "peak_week": 18, "end_week": 48},
    },
    "健康": {
        "keywords": [
            "医疗", "健康", "中医药", "疫苗", "医保",
            "心理健康", "养生", "营养", "健身", "慢性病",
            "互联网医疗", "远程诊疗", "基因检测", "精准医疗", "养老健康",
            "食品安全", "公共卫生", "疫情防控", "医疗改革", "健康中国",
        ],
        "entities": {
            "person": ["钟南山", "张文宏", "李兰娟", "陈薇", "王辰",
                       "乔杰", "葛均波", "张伯礼", "仝小林", "路透"],
            "org": ["国家卫健委", "国家医保局", "中国疾控中心", "世卫组织",
                    "北京协和医院", "四川大学华西医院", "瑞金医院",
                    "阿里健康", "京东健康", "平安好医生"],
            "location": ["北京", "上海", "广州", "武汉", "成都",
                         "杭州", "南京", "西安", "长沙", "郑州"],
        },
        "weibo_templates": [
            "@人民日报 {entity}提醒：{keyword}很重要，大家一定要注意身体💪 #健康第一#",
            "今天去{location}的医院看{keyword}，医生说要注意饮食和作息😂",
            "刚看完{entity}关于{keyword}的科普视频，学到了很多！分享给大家 https://t.cn/K9mP3xQ",
            "{keyword}又上热搜了！{entity}的建议很实用，收藏了👍",
            "在{location}体验了{entity}的{keyword}服务，互联网医疗真的太方便了！",
            "笑死，{entity}说{keyword}最好的药是运动和好心情😂😂",
            "朋友们注意了！{entity}发布{keyword}最新指南，这些习惯要改 #健康科普#",
            "据{entity}研究，{keyword}与生活习惯密切相关，早预防早受益🔥",
            "{entity}在{location}开展{keyword}义诊活动，惠及千名群众👏👏",
            "今天参加了{entity}组织的{keyword}讲座，专家讲得太好了！ #科普讲座#",
            "重磅！{entity}研发的{keyword}新药获批上市，患者福音 https://t.cn/Bn7kP2m",
            "谁懂啊，{keyword}真的不能忽视！{entity}的建议一定要听👍👍",
            "@央视新闻 {entity}解读{keyword}新政策，看病更便宜了",
            "{keyword}+{keyword}，{entity}在{location}的创新模式值得推广 #医疗创新#",
            "在{location}亲身体验了{keyword}远程诊疗，效率真的高，点赞！",
            "刚看到{entity}关于{keyword}的最新论文，医学进步太快了💪💪",
            "实测{entity}推荐的{keyword}养生方法，坚持一个月状态明显好转",
            "今天{entity}在{location}宣布{keyword}新成果，医疗技术再突破🔥",
            "@新华网 {entity}：{keyword}是全民健康的重要基石",
            "从{location}传来消息：{entity}{keyword}项目获国家科技进步奖👏",
        ],
        "news_templates": [
            "据国家卫健委消息，{entity}在{location}召开{keyword}工作会议，部署重点任务",
            "据{entity}发布的数据显示，{keyword}覆盖率已达到92%，成效显著",
            "据悉，{entity}联合多部门出台{keyword}新政策，着力解决看病难问题",
            "业内人士指出，{keyword}技术已进入临床应用阶段，{entity}等机构引领创新",
            "据相关统计，{entity}在{location}实施的{keyword}项目惠及百万群众",
            "据{entity}研究报告，{keyword}领域仍存在资源分配不均问题",
            "从{location}获悉，{entity}创新{keyword}服务模式，获患者广泛好评",
            "据国家医保局消息，{entity}主导的{keyword}纳入医保目录",
            "据报道，{entity}在{location}启动{keyword}专项行动，提升基层医疗水平",
            "据相关人士透露，{entity}将加大对{keyword}领域的科研投入",
            "据权威机构评估，{entity}在{keyword}领域的研究水平位居世界前列",
            "据{location}卫健委消息，{entity}获评{keyword}工作先进典型",
            "据报道，{entity}联合高校开展{keyword}攻关，取得重大突破",
            "据相关统计，{entity}的{keyword}平台已服务超过5000万用户",
            "据业内分析，{keyword}将成为健康中国建设的重要支撑",
        ],
        "lifecycle": {"start_week": 2, "peak_week": 14, "end_week": 50},
    },
    "教育": {
        "keywords": [
            "高考", "中考", "双减", "素质教育", "职业教育",
            "高等教育", "义务教育", "在线教育", "教育公平", "教师",
            "课程改革", "考试招生", "研究生", "留学", "校园",
            "学科竞赛", "教育数字化", "双减政策", "课后服务", "家校共育",
        ],
        "entities": {
            "person": ["顾明远", "潘懋元", "朱永新", "俞敏洪", "张邦鑫",
                       "郑强", "施一公", "颜宁", "饶毅", "丘成桐"],
            "org": ["教育部", "各省教育厅", "清华大学", "北京大学", "浙江大学",
                    "上海交通大学", "复旦大学", "南京大学", "中国科学技术大学",
                    "新东方", "好未来", "中公教育", "中国教育学会"],
            "location": ["北京", "上海", "武汉", "南京", "西安",
                         "广州", "成都", "杭州", "长沙", "合肥"],
        },
        "weibo_templates": [
            "@人民日报 {entity}谈{keyword}：教育是国之大计，必须坚持立德树人💪 #教育强国#",
            "今天{entity}在{location}发表关于{keyword}的演讲，太精彩了👍",
            "重磅！{entity}发布{keyword}新政策，这些变化家长一定要知道 https://t.cn/K9mP3xQ",
            "据{entity}消息，{keyword}改革方案正式出台，影响深远🔥",
            "在{location}参观了{entity}的{keyword}项目，教育创新令人印象深刻👏",
            "笑死，{entity}说{keyword}最重要的是培养兴趣，评论区家长都哭了😂",
            "朋友们注意了！{entity}公布{keyword}最新数据，升学率创新高 #教育资讯#",
            "据{entity}调研，{keyword}实施后学生负担明显减轻，效果显著👍",
            "{entity}在{location}调研{keyword}工作，强调要促进教育公平",
            "今天参加了{entity}的{keyword}公开课，收获满满！ #终身学习#",
            "重磅！{entity}宣布{keyword}新举措，教育资源将更加均衡🔥",
            "@央视新闻 {entity}解读{keyword}政策，这些要点与你息息相关",
            "{keyword}+{keyword}，{entity}在{location}的探索获全国推广 #教育创新#",
            "在{location}亲身体验了{keyword}数字化教学，科技改变教育💪💪",
            "刚看完{entity}关于{keyword}的深度报道，教育改革任重道远...",
            "实测{entity}的{keyword}在线课程，质量很高，免费学习太好了👍",
            "今天{entity}在{location}举办{keyword}论坛，专家学者共话教育未来",
            "@新华网 {entity}：{keyword}是教育公平的重要保障",
            "从{location}传来消息：{entity}{keyword}成果获国家级教学成果奖👏",
        ],
        "news_templates": [
            "据教育部消息，{entity}在{location}召开{keyword}工作会议，部署年度重点任务",
            "据{entity}发布的数据显示，{keyword}覆盖率已达到99%，基本实现全覆盖",
            "据悉，{entity}联合多部门出台{keyword}新政策，着力推进教育公平",
            "业内人士指出，{keyword}改革已进入深水区，{entity}提出系统性方案",
            "据相关统计，{entity}在{location}实施的{keyword}项目惠及百万学生",
            "据{entity}调研报告，{keyword}领域仍存在城乡差距问题",
            "从{location}获悉，{entity}创新{keyword}教学模式，获师生广泛好评",
            "据教育部消息，{entity}主导的{keyword}标准正式实施",
            "据报道，{entity}在{location}启动{keyword}提升专项行动",
            "据相关人士透露，{entity}将加大对{keyword}领域的财政投入",
            "据权威机构评估，{entity}在{keyword}改革方面成效位居全国前列",
            "据{location}教育局消息，{entity}获评{keyword}工作先进典型",
            "据报道，{entity}联合高校开展{keyword}研究，为教育决策提供依据",
            "据相关统计，{entity}的{keyword}平台已服务超过一亿学生",
            "据业内分析，{keyword}将成为未来教育发展的核心方向",
        ],
        "lifecycle": {"start_week": 4, "peak_week": 16, "end_week": 48},
    },
    "环保": {
        "keywords": [
            "碳中和", "碳达峰", "新能源", "光伏", "风电",
            "电动汽车", "垃圾分类", "污染防治", "生态保护", "绿色建筑",
            "节能减排", "清洁能源", "可持续发展", "生物多样性", "气候变化",
            "环保督察", "绿水青山", "双碳目标", "碳排放", "可再生能源",
        ],
        "entities": {
            "person": ["丁仲礼", "解振华", "何建坤", "王金南", "贺克斌",
                       "邹才能", "李俊峰", "马军", "常纪文", "潘家华"],
            "org": ["生态环境部", "国家能源局", "国家发改委", "中国环境科学研究院",
                    "比亚迪", "宁德时代", "隆基绿能", "三峡集团", "中国节能",
                    "大自然保护协会", "WWF", "绿色和平"],
            "location": ["北京", "上海", "深圳", "雄安新区", "鄂尔多斯",
                         "青海", "甘肃", "内蒙古", "云南", "海南"],
        },
        "weibo_templates": [
            "@人民日报 {entity}谈{keyword}：绿水青山就是金山银山💪 #美丽中国#",
            "今天在{location}看到{keyword}的新变化，天更蓝水更清了👍",
            "重磅！{entity}发布{keyword}行动计划，这些目标令人振奋 https://t.cn/K9mP3xQ",
            "据{entity}数据，{keyword}取得显著成效，碳排放持续下降🔥",
            "在{location}体验了{entity}的{keyword}项目，绿色科技太厉害了👏",
            "笑死，{entity}说{keyword}人人有责，从垃圾分类做起😂😂",
            "朋友们注意了！{entity}公布{keyword}最新进展，成绩亮眼 #绿色生活#",
            "据{entity}报告，{keyword}投资连续三年增长，绿色发展势头强劲👍",
            "{entity}在{location}调研{keyword}工作，强调要坚定不移走绿色发展之路",
            "今天参加了{entity}组织的{keyword}志愿活动，为环保出一份力！ #志愿服务#",
            "重磅！{entity}宣布{keyword}新目标，提前实现碳达峰🔥",
            "@央视新闻 {entity}解读{keyword}政策，绿色转型加速推进",
            "{keyword}+{keyword}，{entity}在{location}的创新实践获全球关注 #双碳行动#",
            "在{location}亲身体验了{keyword}绿色出行，电动公交太方便了💪💪",
            "刚看完{entity}关于{keyword}的纪录片，保护地球刻不容缓...",
            "实测{entity}的{keyword}产品，节能减排效果显著，为环保点赞👍",
            "今天{entity}在{location}宣布{keyword}新成果，清洁能源再突破🔥",
            "@新华网 {entity}：{keyword}是中华民族永续发展的根本大计",
            "从{location}传来消息：{entity}{keyword}项目获联合国环境署表彰👏",
        ],
        "news_templates": [
            "据生态环境部消息，{entity}在{location}召开{keyword}工作会议，部署重点任务",
            "据{entity}发布的数据显示，{keyword}目标完成度已达85%，成效显著",
            "据悉，{entity}联合多部门出台{keyword}新政策，加快绿色转型",
            "业内人士指出，{keyword}技术已进入规模化应用阶段，{entity}等企业引领发展",
            "据相关统计，{entity}在{location}实施的{keyword}项目减排效果显著",
            "据{entity}研究报告，{keyword}领域仍存在区域发展不平衡问题",
            "从{location}获悉，{entity}创新{keyword}技术路线，获国际认可",
            "据国家能源局消息，{entity}主导的{keyword}项目正式并网发电",
            "据报道，{entity}在{location}启动{keyword}专项行动，为期两年",
            "据相关人士透露，{entity}将加大对{keyword}领域的投资力度",
            "据权威机构评估，{entity}在{keyword}领域的综合实力位居全球前列",
            "据{location}环保部门消息，{entity}获评{keyword}工作先进典型",
            "据报道，{entity}联合国际机构开展{keyword}合作研究",
            "据相关统计，{entity}的{keyword}装机容量突破100GW",
            "据业内分析，{keyword}将成为实现双碳目标的关键路径",
        ],
        "lifecycle": {"start_week": 6, "peak_week": 22, "end_week": 50},
    },
    "国际": {
        "keywords": [
            "中美关系", "一带一路", "联合国", "G20", "RCEP",
            "国际贸易", "地缘政治", "多边主义", "全球化", "外交",
            "制裁", "军备竞赛", "难民危机", "气候变化合作", "自贸区",
            "国际秩序", "大国博弈", "区域合作", "和平发展", "人类命运共同体",
        ],
        "entities": {
            "person": ["拜登", "马克龙", "朔尔茨", "岸田文雄", "尹锡悦",
                       "普京", "泽连斯基", "莫迪", "特拉斯", "李显龙",
                       "王毅", "布林肯", "沙利文", "耶伦", "拉夫罗夫"],
            "org": ["联合国", "欧盟", "北约", "东盟", "上合组织",
                    "世界银行", "IMF", "WTO", "G7", "金砖国家",
                    "阿拉伯联盟", "非洲联盟", "世卫组织"],
            "location": ["华盛顿", "纽约", "日内瓦", "布鲁塞尔", "莫斯科",
                         "东京", "首尔", "新德里", "伦敦", "巴黎"],
        },
        "weibo_templates": [
            "@人民日报 {entity}就{keyword}发表重要声明，展现大国担当💪 #国际视野#",
            "今天{entity}在{location}就{keyword}问题达成共识，好消息👍",
            "重磅！{entity}宣布{keyword}新决定，国际格局将发生重大变化 https://t.cn/K9mP3xQ",
            "据{entity}消息，{keyword}谈判取得突破性进展🔥",
            "在{location}见证了{keyword}的历史性时刻，感慨万千👏",
            "笑死，{entity}在{keyword}问题上的表态又双标了😂😂",
            "朋友们注意了！{entity}发布{keyword}最新报告，这些数据值得关注 #全球趋势#",
            "据{entity}分析，{keyword}趋势不可逆转，合作共赢是唯一出路👍",
            "{entity}在{location}强调：{keyword}需要各国携手应对",
            "今天{entity}出席{keyword}峰会并发表重要讲话，全场起立鼓掌💪💪",
            "重磅！{entity}签署{keyword}协议，多边主义再进一步🔥",
            "@央视新闻 {entity}解读{keyword}局势，专业分析值得一看",
            "{keyword}+{keyword}，{entity}在{location}的斡旋获国际社会赞誉 #和平使者#",
            "在{location}亲历了{keyword}谈判全过程，外交智慧令人钦佩👏👏",
            "刚看完{entity}关于{keyword}的深度分析，国际形势复杂多变...",
            "实测{entity}的{keyword}预测，准确率很高，值得关注👍",
            "今天{entity}在{location}宣布{keyword}新倡议，构建人类命运共同体🔥",
            "@新华社 {entity}：{keyword}是时代潮流，任何人都无法阻挡",
            "从{location}传来消息：{entity}{keyword}提案获高票通过👏",
        ],
        "news_templates": [
            "据新华社报道，{entity}在{location}就{keyword}问题发表重要讲话",
            "据{entity}消息，{keyword}谈判取得实质性进展，各方达成框架协议",
            "据悉，{entity}在{location}出席{keyword}峰会，提出中国方案",
            "业内人士指出，{keyword}格局正在发生深刻变化，{entity}发挥建设性作用",
            "据相关统计，{entity}推动的{keyword}合作项目已覆盖140多个国家",
            "据联合国消息，{entity}就{keyword}问题提交决议草案",
            "从{location}获悉，{entity}与多国签署{keyword}合作协议",
            "据外交部消息，{entity}就{keyword}局势阐明中方立场",
            "据报道，{entity}在{location}启动{keyword}对话机制",
            "据相关人士透露，{entity}将加大对{keyword}领域的外交投入",
            "据权威机构评估，{entity}在{keyword}事务中的影响力持续提升",
            "据{location}消息，{entity}获评{keyword}合作最佳伙伴",
            "据报道，{entity}联合{keyword}成员国发布联合声明",
            "据相关统计，{entity}的{keyword}倡议获得国际社会广泛响应",
            "据业内分析，{keyword}将成为未来十年国际关系的核心议题",
        ],
        "lifecycle": {"start_week": 8, "peak_week": 25, "end_week": 52},
    },
}

# 微博表情符号池
WEIBO_EMOJIS = ["😂", "🔥", "👍", "💪", "👏", "😭", "❤️", "🎉", "✨", "💯",
                "🚀", "🌟", "📌", "💡", "🎯", "🏆", "📊", "🔗", "📢", "💬"]

# 微博口语化后缀
WEIBO_SUFFIXES = [
    "大家怎么看？", "你怎么看？", "你怎么认为？", "你怎么说？",
    "有同感的吗？", "有人关注吗？", "你们觉得呢？", "同意的举手！",
    "太真实了！", "真的绝了！", "太厉害了！", "必须点赞！",
    "转发让更多人看到", "收藏了！", "学到了！", "涨知识了！",
    "分享给大家", "推荐！", "值得一读", "码住！",
]

# 微博@账号池
WEIBO_ACCOUNTS = [
    "@人民日报", "@央视新闻", "@新华社", "@新华网", "@人民网",
    "@环球时报", "@中国日报", "@光明网", "@经济日报", "@科技日报",
    "@央视财经", "@央视体育", "@健康中国", "@教育小微", "@生态环境部",
    "@财经网", "@澎湃新闻", "@界面新闻", "@第一财经", "@21世纪经济报道",
]

# 微博话题标签池（按话题）
WEIBO_HASHTAGS = {
    "科技": ["#科技创新#", "#人工智能#", "#中国芯#", "#5G时代#", "#科技改变生活#",
             "#黑科技#", "#未来科技#", "#数字化转型#", "#元宇宙#", "#量子计算#"],
    "经济": ["#经济形势#", "#A股#", "#价值投资#", "#宏观经济#", "#消费升级#",
             "#理财心得#", "#金融科技#", "#新经济#", "#高质量发展#", "#共同富裕#"],
    "社会": ["#民生改善#", "#社会治理#", "#乡村振兴#", "#共同富裕#", "#城市生活#",
             "#社区治理#", "#公共服务#", "#数字生活#", "#新市民#", "#社会正能量#"],
    "娱乐": ["#必看推荐#", "#追星日常#", "#下饭综艺#", "#创意无限#", "#票房#",
             "#音乐现场#", "#影视推荐#", "#综艺名场面#", "#明星穿搭#", "#热搜#"],
    "政治": ["#大国担当#", "#外交风云#", "#改革先锋#", "#政策速递#", "#法治中国#",
             "#全面从严治党#", "#基层治理#", "#全过程人民民主#", "#国家安全#", "#中国道路#"],
    "体育": ["#冠军之路#", "#现场直击#", "#体育精神#", "#赛事预测#", "#全民健身#",
             "#奥运精神#", "#中国足球#", "#篮球#", "#马拉松#", "#冬奥#"],
    "健康": ["#健康第一#", "#健康科普#", "#医疗创新#", "#养生#", "#心理健康#",
             "#中医药#", "#疫情防控#", "#健康中国#", "#互联网医疗#", "#全民健身#"],
    "教育": ["#教育强国#", "#教育资讯#", "#教育创新#", "#终身学习#", "#双减#",
             "#高考#", "#素质教育#", "#教育公平#", "#在线教育#", "#教师节#"],
    "环保": ["#美丽中国#", "#绿色生活#", "#双碳行动#", "#碳中和#", "#新能源#",
             "#绿水青山#", "#垃圾分类#", "#节能减排#", "#可持续发展#", "#生物多样性#"],
    "国际": ["#国际视野#", "#全球趋势#", "#和平使者#", "#一带一路#", "#多边主义#",
             "#人类命运共同体#", "#大国博弈#", "#全球化#", "#联合国#", "#G20#"],
}

# 新闻来源
NEWS_SOURCES = [
    "新华社", "人民日报", "央视新闻", "光明日报", "经济日报",
    "中国日报", "科技日报", "工人日报", "中国青年报", "法治日报",
    "新华网", "人民网", "澎湃新闻", "界面新闻", "第一财经",
    "21世纪经济报道", "财新网", "环球时报", "中国新闻网", "中国网",
]

# 新闻来源域名
NEWS_SOURCE_DOMAINS = {
    "新华社": "xinhuanet.com",
    "人民日报": "people.com.cn",
    "央视新闻": "cctv.com",
    "光明日报": "gmw.cn",
    "经济日报": "ce.cn",
    "中国日报": "chinadaily.com.cn",
    "科技日报": "stdaily.com",
    "新华网": "xinhuanet.com",
    "人民网": "people.com.cn",
    "澎湃新闻": "thepaper.cn",
    "界面新闻": "jiemian.com",
    "第一财经": "yicai.com",
    "21世纪经济报道": "21jingji.com",
    "财新网": "caixin.com",
    "环球时报": "huanqiu.com",
    "中国新闻网": "chinanews.com",
    "中国网": "china.com.cn",
}

# 新闻正式用语前缀
NEWS_PREFIXES = [
    "据报道", "据悉", "据相关统计", "据权威机构评估",
    "据相关人士透露", "从{}获悉", "据{}消息",
    "业内人士指出", "据{}发布的数据显示",
    "据{}调研报告", "据{}研究报告",
]

# 新闻正式用语后缀
NEWS_SUFFIXES = [
    "成效显著", "意义重大", "影响深远", "进展顺利",
    "取得突破性进展", "获得广泛好评", "位居全国前列",
    "实现历史性突破", "创历史新高", "达到国际领先水平",
    "填补行业空白", "成为全国标杆", "惠及广大群众",
]

# 概念漂移时间点（周数）
DRIFT_WEEK_1 = 20
DRIFT_WEEK_2 = 36

# 漂移时引入的新兴子话题
DRIFT_NEW_TOPICS = {
    DRIFT_WEEK_1: {
        "科技": "大模型",
        "经济": "数字货币",
        "社会": "新就业形态",
        "娱乐": "AI生成内容",
        "健康": "心理健康",
        "教育": "教育数字化",
        "环保": "碳交易",
        "国际": "AI治理",
    },
    DRIFT_WEEK_2: {
        "科技": "具身智能",
        "经济": "低空经济",
        "社会": "银发经济",
        "娱乐": "短剧",
        "健康": "GLP-1",
        "教育": "AI教育",
        "环保": "新型储能",
        "国际": "太空竞争",
    },
}


# ============================================================
# 工具函数
# ============================================================

def hash_user_id(user_index: int) -> str:
    """生成匿名化的用户ID哈希"""
    raw = f"user_{user_index}_{2023}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def generate_timestamp_for_week(year: int, week: int, rng: random.Random) -> datetime:
    """根据年份和周数生成随机时间戳"""
    # 计算该年第一天的日期
    jan_first = datetime(year, 1, 1)
    # 该周的第一天（周一）
    delta_days = (week - 1) * 7
    week_start = jan_first + timedelta(days=delta_days)
    # 在该周内随机选一个时刻
    random_offset = timedelta(
        days=rng.randint(0, 6),
        hours=rng.randint(0, 23),
        minutes=rng.randint(0, 59),
        seconds=rng.randint(0, 59),
    )
    return week_start + random_offset


def sample_with_weights(items: list, weights: list, rng: random.Random, k: int = 1) -> list:
    """按权重采样"""
    if k == 1:
        return [rng.choices(items, weights=weights, k=1)[0]]
    return rng.choices(items, weights=weights, k=k)


def fill_template(template: str, topic_config: dict, rng: random.Random,
                  topic_name: str = None) -> str:
    """
    填充模板中的占位符：
    - {entity}: 从实体池中随机选取（人名/机构名/地名）
    - {keyword}: 从关键词池中随机选取
    - {location}: 从地名池中随机选取
    - {hashtag}: 从话题标签池中随机选取
    """
    entities = topic_config["entities"]
    keywords = topic_config["keywords"]

    # 收集所有需要替换的占位符
    entity_types = list(entities.keys())

    result = template

    # 替换 {entity} 占位符
    while "{entity}" in result:
        etype = rng.choice(entity_types)
        entity = rng.choice(entities[etype])
        result = result.replace("{entity}", entity, 1)

    # 替换 {keyword} 占位符
    used_keywords = set()
    while "{keyword}" in result:
        available = [k for k in keywords if k not in used_keywords]
        if not available:
            available = keywords
        kw = rng.choice(available)
        used_keywords.add(kw)
        result = result.replace("{keyword}", kw, 1)

    # 替换 {location} 占位符
    while "{location}" in result:
        location = rng.choice(entities["location"])
        result = result.replace("{location}", location, 1)

    # 替换 {hashtag} 占位符
    while "{hashtag}" in result:
        if topic_name and topic_name in WEIBO_HASHTAGS:
            hashtag = rng.choice(WEIBO_HASHTAGS[topic_name])
        else:
            hashtag = "#热门话题#"
        result = result.replace("{hashtag}", hashtag, 1)

    return result


def generate_weibo_text(topic_name: str, topic_config: dict,
                        rng: random.Random) -> Tuple[str, List[str]]:
    """生成一条微博文本及其关键词"""
    template = rng.choice(topic_config["weibo_templates"])
    text = fill_template(template, topic_config, rng, topic_name=topic_name)

    # 提取文本中实际使用的关键词
    used_keywords = []
    for kw in topic_config["keywords"]:
        if kw in text and kw not in used_keywords:
            used_keywords.append(kw)

    # 补充关键词到3-8个
    if len(used_keywords) < 3:
        additional = rng.sample(
            [k for k in topic_config["keywords"] if k not in used_keywords],
            min(3 - len(used_keywords), len(topic_config["keywords"]) - len(used_keywords))
        )
        used_keywords.extend(additional)

    # 随机截取3-8个关键词
    used_keywords = used_keywords[:rng.randint(3, min(8, len(used_keywords)))]

    return text, used_keywords


def generate_news_text(topic_name: str, topic_config: dict,
                       rng: random.Random) -> Tuple[str, str, List[str]]:
    """生成一条新闻文本（标题+正文）及其编辑关键词"""
    # 生成标题
    title_template = rng.choice(topic_config["news_templates"])
    title = fill_template(title_template, topic_config, rng, topic_name=topic_name)

    # 生成正文（组合多个模板片段 + 正式用语扩展）
    body_parts = []

    # 选取1-2个新闻模板作为正文基础
    num_body_templates = rng.randint(1, 2)
    body_templates = rng.sample(topic_config["news_templates"], min(num_body_templates, len(topic_config["news_templates"])))
    for bt in body_templates:
        filled = fill_template(bt, topic_config, rng, topic_name=topic_name)
        body_parts.append(filled)

    # 添加正式用语扩展段落
    entity_types = list(topic_config["entities"].keys())
    org = rng.choice(topic_config["entities"]["org"])
    location = rng.choice(topic_config["entities"]["location"])
    keyword = rng.choice(topic_config["keywords"])

    expansion_templates = [
        f"据{org}相关负责人介绍，近年来在{keyword}领域持续加大投入力度，"
        f"累计投入资金超过百亿元，取得了一系列重要成果。下一步，"
        f"将继续深化{keyword}领域改革创新，推动产业高质量发展。",

        f"记者在{location}采访了解到，{keyword}相关产业近年来发展迅速，"
        f"已形成较为完整的产业链条。当地政府高度重视{keyword}产业发展，"
        f"出台了一系列扶持政策，为企业发展创造了良好环境。",

        f"专家表示，{keyword}领域的快速发展得益于政策支持、技术进步和市场需求的"
        f"多重驱动。展望未来，随着各项改革措施的深入推进，{keyword}产业"
        f"将迎来更加广阔的发展空间。",

        f"据{org}发布的研究报告显示，{keyword}市场规模持续扩大，"
        f"预计未来五年将保持年均15%以上的增速。报告同时指出，"
        f"行业仍面临人才短缺、核心技术攻关等挑战。",

        f"从{location}传来的最新消息，{keyword}示范项目已取得阶段性成果，"
        f"各项指标均达到预期目标。项目负责人表示，将在总结经验的基础上，"
        f"进一步扩大试点范围，加快推广步伐。",
    ]

    body_parts.append(rng.choice(expansion_templates))

    # 组合正文
    body = "\n\n".join(body_parts)

    # 提取关键词
    full_text = title + " " + body
    used_keywords = []
    for kw in topic_config["keywords"]:
        if kw in full_text and kw not in used_keywords:
            used_keywords.append(kw)

    # 编辑关键词：4-8个
    if len(used_keywords) < 4:
        additional = rng.sample(
            [k for k in topic_config["keywords"] if k not in used_keywords],
            min(4 - len(used_keywords), len(topic_config["keywords"]) - len(used_keywords))
        )
        used_keywords.extend(additional)

    used_keywords = used_keywords[:rng.randint(4, min(8, len(used_keywords)))]

    return title, body, used_keywords


# ============================================================
# 数据集生成器
# ============================================================

class DKNetDatasetGenerator:
    """DKNet 高质量数据集生成器"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.topic_names = list(TOPIC_DEFINITIONS.keys())

    def _compute_topic_weights(self, week: int, topic_name: str) -> float:
        """
        根据话题生命周期计算指定周的话题权重（用于采样分布）
        使用正态分布围绕峰值周
        """
        lifecycle = TOPIC_DEFINITIONS[topic_name]["lifecycle"]
        start = lifecycle["start_week"]
        peak = lifecycle["peak_week"]
        end = lifecycle["end_week"]

        if week < start or week > end:
            return 0.0

        # 正态分布：以peak为中心，sigma根据生命周期宽度调整
        width = (end - start) / 2.0
        sigma = max(width / 2.5, 2.0)
        weight = np.exp(-0.5 * ((week - peak) / sigma) ** 2)
        return float(weight)

    def _sample_topic_for_week(self, week: int) -> str:
        """根据周数按权重采样话题"""
        weights = []
        for tn in self.topic_names:
            w = self._compute_topic_weights(week, tn)
            weights.append(w)

        # 归一化
        total = sum(weights)
        if total == 0:
            return self.rng.choice(self.topic_names)
        weights = [w / total for w in weights]

        return self.rng.choices(self.topic_names, weights=weights, k=1)[0]

    def _sample_week_count(self, topic_name: str, total_weeks: int = 52) -> Dict[int, int]:
        """
        为指定话题生成各周的样本数量分布
        返回 {week: count} 字典
        """
        lifecycle = TOPIC_DEFINITIONS[topic_name]["lifecycle"]
        start = lifecycle["start_week"]
        peak = lifecycle["peak_week"]
        end = lifecycle["end_week"]

        counts = {}
        for w in range(start, end + 1):
            weight = self._compute_topic_weights(w, topic_name)
            # 基础计数 + 权重加成
            base = self.np_rng.poisson(50)
            counts[w] = max(0, int(base * weight * 3))

        return counts

    def generate_d1_weibo(self, num_samples: int = 150000,
                          start_year: int = 2023) -> List[Dict]:
        """
        生成 D1 微博时序数据集
        时间范围：2023年1月-12月（52周）
        """
        print(f"[D1] 开始生成微博时序数据集，目标样本数: {num_samples}")

        records = []

        # 计算每个话题的周分布
        topic_week_counts = {}
        for tn in self.topic_names:
            topic_week_counts[tn] = self._sample_week_count(tn)

        # 归一化到目标样本数
        total_planned = sum(
            c for tc in topic_week_counts.values() for c in tc.values()
        )
        if total_planned > 0:
            scale = num_samples / total_planned
        else:
            scale = 1.0

        # 按话题和周生成
        for topic_name in self.topic_names:
            topic_config = TOPIC_DEFINITIONS[topic_name]
            week_counts = topic_week_counts[topic_name]

            for week, raw_count in week_counts.items():
                count = max(1, int(raw_count * scale))
                for _ in range(count):
                    # 生成文本
                    text, keywords = generate_weibo_text(topic_name, topic_config, self.rng)

                    # 生成时间戳
                    timestamp = generate_timestamp_for_week(start_year, week, self.rng)

                    # 生成用户ID哈希
                    user_id = hash_user_id(self.rng.randint(1, 50000))

                    # 生成互动数据（转发/评论/点赞）
                    # 大部分微博互动量较低，少数热门微博互动量高（幂律分布）
                    base_engagement = self.np_rng.lognormal(mean=2, sigma=1.5)
                    reposts = max(0, int(base_engagement * self.rng.uniform(0.3, 1.0)))
                    comments = max(0, int(base_engagement * self.rng.uniform(0.2, 0.8)))
                    likes = max(0, int(base_engagement * self.rng.uniform(0.5, 2.0)))

                    # 确定领域标签（主话题 + 可能的交叉话题）
                    domain = topic_name
                    if self.rng.random() < 0.1:
                        # 10%概率标记为交叉领域
                        other = self.rng.choice([t for t in self.topic_names if t != topic_name])
                        domain = f"{topic_name}/{other}"

                    records.append({
                        "text": text,
                        "keywords": keywords,
                        "timestamp": timestamp.isoformat(),
                        "user_id_hash": user_id,
                        "topic": topic_name,
                        "domain": domain,
                        "engagement": {
                            "reposts": reposts,
                            "comments": comments,
                            "likes": likes,
                        },
                        "week": week,
                    })

        # 如果记录数超过目标，随机采样
        if len(records) > num_samples:
            self.rng.shuffle(records)
            records = records[:num_samples]

        # 按时间排序
        records.sort(key=lambda x: x["timestamp"])

        # 移除辅助字段
        for r in records:
            r.pop("week", None)

        print(f"[D1] 生成完成，实际样本数: {len(records)}")
        return records

    def generate_d2_news(self, num_samples: int = 50000) -> List[Dict]:
        """
        生成 D2 新闻门户数据集
        时间范围：2019-2022（训练）+ 2023（测试）
        """
        print(f"[D2] 开始生成新闻门户数据集，目标样本数: {num_samples}")

        records = []

        # 训练集：2019-2022，占80%；测试集：2023，占20%
        train_count = int(num_samples * 0.8)
        test_count = num_samples - train_count

        # 训练集年份
        train_years = [2019, 2020, 2021, 2022]
        # 测试集年份
        test_years = [2023]

        # 为每个年份分配样本数
        year_counts = {}
        for y in train_years:
            year_counts[y] = train_count // len(train_years)
        # 余数分配给最后一个训练年
        remainder = train_count - sum(year_counts.values())
        year_counts[train_years[-1]] += remainder
        year_counts[2023] = test_count

        for year, count in year_counts.items():
            for _ in range(count):
                # 采样话题
                topic_name = self.rng.choice(self.topic_names)
                topic_config = TOPIC_DEFINITIONS[topic_name]

                # 生成新闻文本
                title, body, keywords = generate_news_text(topic_name, topic_config, self.rng)

                # 生成时间戳
                month = self.rng.randint(1, 12)
                day = self.rng.randint(1, 28)
                hour = self.rng.choice([7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20])
                minute = self.rng.randint(0, 59)
                try:
                    timestamp = datetime(year, month, day, hour, minute)
                except ValueError:
                    timestamp = datetime(year, month, 1, hour, minute)

                # 选择新闻来源
                source = self.rng.choice(NEWS_SOURCES)

                records.append({
                    "text": body,
                    "title": title,
                    "keywords": keywords,
                    "timestamp": timestamp.isoformat(),
                    "source": source,
                    "category": topic_name,
                    "split": "train" if year in train_years else "test",
                })

        # 按时间排序
        records.sort(key=lambda x: x["timestamp"])

        print(f"[D2] 生成完成，实际样本数: {len(records)}")
        print(f"[D2] 训练集: {sum(1 for r in records if r['split'] == 'train')}")
        print(f"[D2] 测试集: {sum(1 for r in records if r['split'] == 'test')}")
        return records

    def apply_concept_drift_d1(self, records: List[Dict]) -> List[Dict]:
        """
        对 D1 数据集应用概念漂移
        在第20周和第36周引入新话题关键词
        """
        print("[D1] 应用概念漂移...")

        # 重新计算每条记录的周数（基于时间戳）
        for record in records:
            ts = datetime.fromisoformat(record["timestamp"])
            # 计算2023年的周数
            jan_first = datetime(2023, 1, 1)
            week = ((ts - jan_first).days // 7) + 1
            record["_week"] = week

        drift_count = 0
        for record in records:
            week = record.get("_week", 0)

            for drift_week, drift_topics in DRIFT_NEW_TOPICS.items():
                if week >= drift_week and record["topic"] in drift_topics:
                    # 30%概率注入漂移关键词
                    if self.rng.random() < 0.3:
                        new_keyword = drift_topics[record["topic"]]
                        if new_keyword not in record["keywords"]:
                            # 替换一个随机关键词
                            if record["keywords"]:
                                idx = self.rng.randint(0, len(record["keywords"]) - 1)
                                record["keywords"][idx] = new_keyword
                            else:
                                record["keywords"].append(new_keyword)
                            drift_count += 1

            # 清理辅助字段
            record.pop("_week", None)

        print(f"[D1] 概念漂移应用完成，影响样本数: {drift_count}")
        return records

    def apply_concept_drift_d2(self, records: List[Dict]) -> List[Dict]:
        """
        对 D2 数据集应用概念漂移
        2023年（测试集）的数据中注入新话题关键词
        """
        print("[D2] 应用概念漂移...")

        drift_count = 0
        for record in records:
            if record["split"] != "test":
                continue

            ts = datetime.fromisoformat(record["timestamp"])
            # 2023年的周数
            jan_first = datetime(2023, 1, 1)
            week = ((ts - jan_first).days // 7) + 1

            for drift_week, drift_topics in DRIFT_NEW_TOPICS.items():
                if week >= drift_week and record["category"] in drift_topics:
                    if self.rng.random() < 0.3:
                        new_keyword = drift_topics[record["category"]]
                        if new_keyword not in record["keywords"]:
                            if record["keywords"]:
                                idx = self.rng.randint(0, len(record["keywords"]) - 1)
                                record["keywords"][idx] = new_keyword
                            else:
                                record["keywords"].append(new_keyword)
                            drift_count += 1

        print(f"[D2] 概念漂移应用完成，影响样本数: {drift_count}")
        return records

    def save_json(self, records: List[Dict], filepath: Path):
        """保存为JSON格式"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {filepath} ({len(records)} 条记录)")

    def save_csv(self, records: List[Dict], filepath: Path, dataset_type: str):
        """保存为CSV格式"""
        if not records:
            print(f"  警告: 无数据可保存到 {filepath}")
            return

        # 展平记录
        flat_records = []
        for r in records:
            flat = {}
            for key, value in r.items():
                if key == "keywords":
                    flat[key] = "|".join(value)
                elif key == "engagement" and isinstance(value, dict):
                    for ek, ev in value.items():
                        flat[f"engagement_{ek}"] = ev
                elif key == "split" and dataset_type == "d1":
                    # D1不需要split字段
                    continue
                else:
                    flat[key] = value
            flat_records.append(flat)

        # 写入CSV
        fieldnames = list(flat_records[0].keys())
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_records)

        print(f"  已保存: {filepath} ({len(flat_records)} 条记录)")

    def generate_statistics(self, d1_records: List[Dict],
                            d2_records: List[Dict]) -> Dict:
        """生成数据集统计信息"""
        stats = {
            "generated_at": datetime.now().isoformat(),
            "seed": self.seed,
            "d1_weibo_temporal": {
                "total_samples": len(d1_records),
                "time_range": {
                    "start": d1_records[0]["timestamp"] if d1_records else None,
                    "end": d1_records[-1]["timestamp"] if d1_records else None,
                },
                "avg_text_length": float(np.mean([len(r["text"]) for r in d1_records])) if d1_records else 0,
                "min_text_length": min(len(r["text"]) for r in d1_records) if d1_records else 0,
                "max_text_length": max(len(r["text"]) for r in d1_records) if d1_records else 0,
                "avg_keywords_per_sample": float(np.mean([len(r["keywords"]) for r in d1_records])) if d1_records else 0,
                "unique_keywords": len(set(kw for r in d1_records for kw in r["keywords"])) if d1_records else 0,
                "unique_users": len(set(r["user_id_hash"] for r in d1_records)) if d1_records else 0,
                "topic_distribution": dict(Counter(r["topic"] for r in d1_records)) if d1_records else {},
                "avg_engagement": {
                    "reposts": float(np.mean([r["engagement"]["reposts"] for r in d1_records])) if d1_records else 0,
                    "comments": float(np.mean([r["engagement"]["comments"] for r in d1_records])) if d1_records else 0,
                    "likes": float(np.mean([r["engagement"]["likes"] for r in d1_records])) if d1_records else 0,
                },
            },
            "d2_news_portal": {
                "total_samples": len(d2_records),
                "train_samples": sum(1 for r in d2_records if r.get("split") == "train"),
                "test_samples": sum(1 for r in d2_records if r.get("split") == "test"),
                "time_range": {
                    "start": d2_records[0]["timestamp"] if d2_records else None,
                    "end": d2_records[-1]["timestamp"] if d2_records else None,
                },
                "avg_text_length": float(np.mean([len(r["text"]) for r in d2_records])) if d2_records else 0,
                "avg_title_length": float(np.mean([len(r["title"]) for r in d2_records])) if d2_records else 0,
                "avg_keywords_per_sample": float(np.mean([len(r["keywords"]) for r in d2_records])) if d2_records else 0,
                "unique_keywords": len(set(kw for r in d2_records for kw in r["keywords"])) if d2_records else 0,
                "unique_sources": len(set(r["source"] for r in d2_records)) if d2_records else 0,
                "category_distribution": dict(Counter(r["category"] for r in d2_records)) if d2_records else {},
                "source_distribution": dict(Counter(r["source"] for r in d2_records)) if d2_records else {},
            },
            "concept_drift": {
                "drift_points": [DRIFT_WEEK_1, DRIFT_WEEK_2],
                "drift_week_1_new_topics": DRIFT_NEW_TOPICS[DRIFT_WEEK_1],
                "drift_week_2_new_topics": DRIFT_NEW_TOPICS[DRIFT_WEEK_2],
            },
        }

        return stats


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DKNet 高质量数据集生成脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_datasets.py --quick
  python generate_datasets.py --output_dir ./datasets --d1_samples 150000 --d2_samples 50000
  python generate_datasets.py --seed 123 --output_dir ./test_data
        """,
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认为脚本所在目录下的 datasets 子目录）",
    )
    parser.add_argument(
        "--d1_samples", type=int, default=150000,
        help="D1 微博数据集样本数（默认: 150000）",
    )
    parser.add_argument(
        "--d2_samples", type=int, default=50000,
        help="D2 新闻数据集样本数（默认: 50000）",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式：D1=100条，D2=80条",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认: 42）",
    )

    args = parser.parse_args()

    # 快速模式
    if args.quick:
        args.d1_samples = 100
        args.d2_samples = 80
        print("[快速模式] D1=100条, D2=80条")

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)

    # 初始化生成器
    generator = DKNetDatasetGenerator(seed=args.seed)

    # 生成 D1 微博数据集
    print()
    d1_records = generator.generate_d1_weibo(num_samples=args.d1_samples)

    # 生成 D2 新闻数据集
    print()
    d2_records = generator.generate_d2_news(num_samples=args.d2_samples)

    # 应用概念漂移
    print()
    d1_records = generator.apply_concept_drift_d1(d1_records)
    d2_records = generator.apply_concept_drift_d2(d2_records)

    # 保存数据集
    print()
    print("[保存] 开始保存数据集...")

    # D1 JSON
    generator.save_json(d1_records, output_dir / "d1_weibo_temporal.json")
    # D1 CSV
    generator.save_csv(d1_records, output_dir / "d1_weibo_temporal.csv", "d1")

    # D2 JSON
    generator.save_json(d2_records, output_dir / "d2_news_portal.json")
    # D2 CSV
    generator.save_csv(d2_records, output_dir / "d2_news_portal.csv", "d2")

    # 生成统计信息
    print()
    print("[统计] 生成数据集统计信息...")
    stats = generator.generate_statistics(d1_records, d2_records)
    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {stats_path}")

    # 打印统计摘要
    print()
    print("=" * 60)
    print("数据集生成完成！统计摘要：")
    print("-" * 60)
    d1s = stats["d1_weibo_temporal"]
    d2s = stats["d2_news_portal"]
    print(f"D1 微博时序数据集:")
    print(f"  样本数: {d1s['total_samples']}")
    print(f"  时间范围: {d1s['time_range']['start']} ~ {d1s['time_range']['end']}")
    print(f"  平均文本长度: {d1s['avg_text_length']:.1f} 字符")
    print(f"  文本长度范围: {d1s['min_text_length']} ~ {d1s['max_text_length']} 字符")
    print(f"  平均关键词数: {d1s['avg_keywords_per_sample']:.1f}")
    print(f"  唯一关键词数: {d1s['unique_keywords']}")
    print(f"  唯一用户数: {d1s['unique_users']}")
    print(f"  平均互动 - 转发: {d1s['avg_engagement']['reposts']:.1f}, "
          f"评论: {d1s['avg_engagement']['comments']:.1f}, "
          f"点赞: {d1s['avg_engagement']['likes']:.1f}")
    print(f"  话题分布:")
    for topic, count in sorted(d1s["topic_distribution"].items()):
        ratio = count / d1s["total_samples"] * 100
        print(f"    {topic}: {count} ({ratio:.1f}%)")

    print()
    print(f"D2 新闻门户数据集:")
    print(f"  样本数: {d2s['total_samples']} (训练: {d2s['train_samples']}, 测试: {d2s['test_samples']})")
    print(f"  时间范围: {d2s['time_range']['start']} ~ {d2s['time_range']['end']}")
    print(f"  平均正文长度: {d2s['avg_text_length']:.1f} 字符")
    print(f"  平均标题长度: {d2s['avg_title_length']:.1f} 字符")
    print(f"  平均关键词数: {d2s['avg_keywords_per_sample']:.1f}")
    print(f"  唯一关键词数: {d2s['unique_keywords']}")
    print(f"  唯一来源数: {d2s['unique_sources']}")
    print(f"  类别分布:")
    for cat, count in sorted(d2s["category_distribution"].items()):
        ratio = count / d2s["total_samples"] * 100
        print(f"    {cat}: {count} ({ratio:.1f}%)")

    print()
    print(f"概念漂移:")
    print(f"  漂移点: 第{DRIFT_WEEK_1}周, 第{DRIFT_WEEK_2}周")
    print(f"  第{DRIFT_WEEK_1}周新话题: {', '.join(DRIFT_NEW_TOPICS[DRIFT_WEEK_1].values())}")
    print(f"  第{DRIFT_WEEK_2}周新话题: {', '.join(DRIFT_NEW_TOPICS[DRIFT_WEEK_2].values())}")
    print("=" * 60)
    print(f"所有文件已保存到: {output_dir}")


if __name__ == "__main__":
    main()
