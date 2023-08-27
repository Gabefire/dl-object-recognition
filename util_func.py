import re


def find_date_fields(text_list: [str]) -> [str]:
    re_list = []
    for text in text_list:
        if re.search("([0-9]){1,2}\/([0-9]){1,2}\/([0-9]){2,4}", text):
            text.strip()
            text = re.search(r"([0-9]){1,2}\/([0-9]){1,2}\/([0-9]){2,4}", text)
            re_list.append(text.group())
    return re_list
