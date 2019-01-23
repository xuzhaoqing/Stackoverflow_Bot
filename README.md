# StackOverflow Assistant Project

标签（空格分隔）： NLP

---

这个`Project`是`Coursera`上 [National Research University Higher School of Economics](https://en.wikipedia.org/wiki/https://en.wikipedia.org/wiki/National_Research_University_%E2%80%93_Higher_School_of_Economics)的 [Advanced Machine Learning](https://www.coursera.org/specializations/aml) 的第六门课 [Natural Language Processing](https://www.coursera.org/learn/language-processing) 的`week5`的`Project`，整个专项课程的所有代码均可以在[Github](https://github.com/hse-aml)上找到，包括这个`Project`的代码

## Outcome
它的目的是在`Twitter`上建立一个可以简单与人交流并回答具体编程问题的机器人，并且主要用到了第一周和第三周的知识点
 
## AWS配置
这里面有一个特别有毒的配置就是当你在云上安装好`Docker`之后，你会遇到这种情况：
> **Note:** If you are getting an error "Got permission denied while trying to connect to the Docker daemon socket...", you need to add
> current user to the docker group: 
>
> sh sudo usermod -a -G docker
> $USER sudo service docker restart

但是这个`$USER`究竟是什么呢？我最初以为是Docker的username，最后才发现是这个云linux的username，对于一个ubuntu的EC2 image来说，这个`$USER`是`ubuntu`。

## Bot的“Hello World”
问题来了，我在配置好了之后开始跟`bot`对话，可是却没有回应，如下所示：

> $ sh
> start_main.sh
> Ready to talk!
> [my input but got no response]

看了论坛才知道原来你只要打开`main_bot.py`就好了，然后要在`Telegram`的`app`里跟你的`Project Bot`聊天

## 识别意图
```python
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

```
这一步我们要做的内容是当用户发出一个`programming-related questions`，`bot`可以自动辨别
### 数据预处理
```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('',text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split(' ') if word and word not in STOPWORDS]) # delete stopwords from text
    return text
    
def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""
    
    # Train a vectorizer on X_train data.
    # Transform X_train and X_test data.
    
    # Pickle the trained vectorizer to 'vectorizer_path'
    # Don't forget to open the file in writing bytes mode.
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, ngram_range=(1,2),token_pattern=re.compile('\S+')) ####### YOUR CODE HERE #######
    
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    pickle.dump(tfidf_vectorizer, vectorizer_path)
    return X_train, X_test
```
基本照抄week1 作业，值得注意的是这里我们使用了[pickle.dump()](https://blog.csdn.net/gdkyxy2013/article/details/80495353)用法，将训练得到的`TFIDF` model保存下来。

```python
sample_size = 200000
dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)
```
从文件中抽出`200000`个`sample`数据来，数据形式如下：
```python
dialogue_df.head()
```
| index	| text |	tag |
|  ---     ---       ---
| 82925	| Donna, you are a muffin. |	dialogue |
| 48774	| He was here last night till about two o'clock....	| dialogue |
| 55394	| All right, then make an appointment with her s...	| dialogue |
| 90806	| Hey, what is this-an interview? We're supposed...	| dialogue |
| 107758 |	Yeah. He's just a friend of mine I was trying ... | dialogue |
```python
stackoverflow_df.head()
```

| index | post_id	| title	| tag
| ---
| 2168983 |	43837842 | efficient algorithm compose valid expressions ... | python
| 1084095 | 15747223 | basic thread program fail clang pass g++	| c_cpp
| 1049020 |	15189594 | link scroll top working	| javascript | 
| 200466 | 3273927 | possible implement ping windows phone 7 |	c#
| 1200249	| 17684551	| glsl normal mapping issue	| c_cpp

```python
from utils import text_prepare

dialogue_df['text'] = [text_prepare(text) for text in dialogue_df['text'] ]######### YOUR CODE HERE #############

stackoverflow_df['title'] = [text_prepare(title) for title in stackoverflow_df['title']]######### YOUR CODE HERE #############
```
进行数据清洗工作

### Intent Recognition
我们将这两个数据集合为一个，标签分为两类：`dialogue`和`stackoverflow`，然后用`TF-IDF`编码：
```python
from sklearn.model_selection import train_test_split

```

    ValueError: setting an array element with a sequence.
出现这个错误是因为`y_train`是一个`list`而不是一个`array`，

