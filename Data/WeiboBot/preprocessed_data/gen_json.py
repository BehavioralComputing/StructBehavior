import os
import json
import re
import pandas as pd
from bs4 import BeautifulSoup


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    for img in soup.find_all("img"):
        if img.has_attr("alt"):
            img.replace_with(img["alt"])

    text = soup.get_text(separator="").strip()

    return remove_invisible_chars(text)


def parse_user_file(filepath):
    user_data = {}
    posts = []
    current_post = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    user_data["id"] = os.path.splitext(os.path.basename(filepath))[0].replace(
        "weibo", ""
    )

    avg_like = 0
    avg_comment = 0
    avg_repost = 0
    avg_image_count = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("微博昵称："):
            user_data["nickname"] = line.replace("微博昵称：", "")
        elif line.startswith("微博主页地址："):
            user_data["profile_url"] = line.replace("微博主页地址：", "")
        elif line.startswith("微博头像地址："):
            user_data["avatar_url"] = line.replace("微博头像地址：", "")
        elif line.startswith("是否认证："):
            user_data["verified"] = line.replace("是否认证：", "") == "True"
        elif line.startswith("微博说明："):
            user_data["description"] = line.replace("微博说明：", "")
        elif line.startswith("关注人数："):
            user_data["follows"] = int(line.replace("关注人数：", ""))
        elif line.startswith("粉丝数："):
            user_data["followers"] = int(line.replace("粉丝数：", ""))
        elif line.startswith("性别："):
            user_data["gender"] = line.replace("性别：", "")
        elif line.startswith("微博等级："):
            user_data["level"] = int(line.replace("微博等级：", ""))
        elif line.startswith("----第") and "微博----" in line:
            if current_post:
                posts.append(current_post)
            current_post = {}
        elif current_post is not None:
            if line.startswith("微博地址："):
                current_post["url"] = line.replace("微博地址：", "")
            elif line.startswith("发布时间："):
                current_post["time"] = line.replace("发布时间：", "").strip()

                if "昨天" in line or "前" in line or "刚刚" in line:
                    print(
                        "Warning: Detected '昨天' or '前' or '刚刚' in time, assuming data is from 2020-07-04"
                    )
                    print(f"Original time: {current_post['time']}")
                    current_post["time"] = "2020-07-04"

                elif len(current_post["time"]) == 5:
                    current_post["time"] = f"2020-{current_post['time']}"
            elif line.startswith("微博内容："):
                raw_content = line.replace("微博内容：", "")
                while i + 1 < len(lines) and not lines[i + 1].startswith("点赞数："):
                    i += 1
                    raw_content += lines[i].strip()

                current_post["raw_content"] = raw_content
                current_post["cleaned_content"] = clean_html(raw_content)
            elif line.startswith("点赞数："):
                current_post["like"] = int(line.replace("点赞数：", ""))
                avg_like += current_post["like"]
            elif line.startswith("评论数："):
                current_post["comment"] = int(line.replace("评论数：", ""))
                avg_comment += current_post["comment"]
            elif line.startswith("转发数："):
                current_post["repost"] = int(line.replace("转发数：", ""))
                avg_repost += current_post["repost"]
            elif line.startswith("图片数："):
                current_post["image_count"] = int(line.replace("图片数：", ""))
                avg_image_count += current_post["image_count"]
        i += 1

    if current_post:
        posts.append(current_post)

    user_data["posts"] = posts

    user_data["avg_like"] = avg_like / len(posts) if posts else 0
    user_data["avg_comment"] = avg_comment / len(posts) if posts else 0
    user_data["avg_repost"] = avg_repost / len(posts) if posts else 0
    user_data["avg_image_count"] = avg_image_count / len(posts) if posts else 0

    return user_data


def parse_all_users(folder_path):
    all_users = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            try:
                user_data = parse_user_file(filepath)
                all_users.append(user_data)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return all_users


def remove_invisible_chars(text):
    return re.sub(r"[\u200e\u200c\u200b]", "", text)


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(BASE_DIR, "../raw/final_data")
    label_file = os.path.join(BASE_DIR, "../raw/bot_label.xlsx")
    output_labeled = os.path.join(BASE_DIR, "weibo_labeled.json")
    output_support = os.path.join(BASE_DIR, "weibo_support.json")

    if os.path.exists(output_labeled) and os.path.exists(output_support):
        print("Labeled and support files already exist. Skipping parsing.")
        return

    df = pd.read_excel(label_file)
    uid_to_label = dict(zip(df["uid"].astype(str), df["botornot"]))

    all_users = parse_all_users(folder)

    labeled_users = []
    support_users = []

    for user in all_users:
        uid = user["id"]
        if uid in uid_to_label:
            user["label"] = uid_to_label[uid]
            labeled_users.append(user)
        else:
            support_users.append(user)

    with open(output_labeled, "w", encoding="utf-8") as f:
        json.dump(labeled_users, f, ensure_ascii=False, indent=2)

    with open(output_support, "w", encoding="utf-8") as f:
        json.dump(support_users, f, ensure_ascii=False, indent=2)

    print(
        f"There are {len(labeled_users)} labeled user, {len(support_users)} support user."
    )


if __name__ == "__main__":
    main()
