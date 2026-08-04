import json
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
# 配置客户端（填入你的真实密钥）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 中文句子列表（80个）
sentences = [
    "今天天气很好", "我喜欢编程", "明天会更好", "这本书很有趣", "我想去旅行",
    "他是一名老师", "她在学习英语", "我们在吃饭", "他们会来的", "你应该休息",
    "学习很重要", "工作很努力", "时间很宝贵", "友谊很珍贵", "家庭很温暖",
    "人工智能很强大", "大数据很有用", "云计算很便捷", "互联网很普及", "科技改变生活",
    "这个菜很好吃", "那家餐厅很棒", "我喜欢吃水果", "她喜欢喝咖啡", "我们喜欢吃面",
    "我想去北京", "我想去上海", "我想去广州", "我想去杭州", "我想去成都",
    "长城很壮观", "故宫很宏伟", "西湖很美", "外滩很漂亮", "东方明珠很高",
    "坚持就是胜利", "失败是成功之母", "团结就是力量", "知识就是力量", "诚信是做人之本",
    "我很高兴见到你", "很高兴认识你", "见到你很高兴", "很高兴和你交流", "很高兴和你合作",
    "这个方案很好", "那个计划很棒", "这个想法很新颖", "那个建议很实用", "这个方案很可行",
    "每天学习一点点", "进步需要努力", "成功源于坚持", "失败不可怕", "重新站起来",
    "大海很辽阔", "天空很蓝", "星星很亮", "月亮很美", "太阳很温暖",
    "春天来了", "夏天很热", "秋天很凉", "冬天很冷", "四季分明",
    "朋友很重要", "家人最亲", "老师很负责", "同学很友好", "同事很合作",
    "工作要认真", "学习要勤奋", "生活要快乐", "健康要珍惜", "时间要利用",
    "梦想很美好", "现实很骨感", "努力很值得", "付出有回报", "未来很可期",
]

print(f"📊 共 {len(sentences)} 个中文句子，开始翻译...")

data = []
for i, ch in enumerate(sentences):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "把中文翻译成英文，只输出翻译结果，不要解释"},
                {"role": "user", "content": ch}
            ],
            temperature=0.3
        )
        en = response.choices[0].message.content.strip()
        data.append({"instruction": "翻译成英文", "input": ch, "output": en})

        if (i + 1) % 10 == 0:
            print(f"  ✅ 已完成 {i + 1}/{len(sentences)} 条")
    except Exception as e:
        print(f"  ❌ 翻译失败：{ch}, 错误：{e}")
        data.append({"instruction": "翻译成英文", "input": ch, "output": ch})

# 保存数据
with open("translation_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 完成！共生成 {len(data)} 条翻译数据，保存到 translation_data.json")