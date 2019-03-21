import jieba
import random
import glob


def load_stop_words(path):
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        return [line.strip() for line in f]


def load_content(path):
    with open(path, 'r', encoding='gbk', errors='ignore') as f:
        content = ''
        for line in f:
            line = line.strip()
            content += line
    return content


def get_TF(words, topK=10):
    tf_dic = {}
    for w in split_words:
        tf_dic[w] = tf_dic.get(w, 0) + 1
    topK_words = sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:topK]
    return topK_words


if __name__ == '__main__':
    files = glob.glob('D:/ML/BookSourceCode/learning-nlp/chapter-3/data/news/C000013/*.txt')
    corpus = [load_content(x) for x in files]

    stop_words = load_stop_words('D:/ML/BookSourceCode/learning-nlp/chapter-3/data/stop_words.utf8')

    split_words = list(jieba.cut(corpus[0]))
    # 去除停用词
    split_words = [w for w in split_words if w not in stop_words]
    print('分词效果：' + '/ '.join(split_words))

    # 统计高频词
    top10_words = get_TF(split_words, topK=10)
    print(top10_words)
