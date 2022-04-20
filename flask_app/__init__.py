#FLASK_APP=flask_app flask run
from flask import Flask, render_template
import pickle, pandas
import numpy as np
# 코사인유사도 계산
from sklearn.feature_extraction.text import CountVectorizer  # 피체 벡터화
from sklearn.metrics.pairwise import cosine_similarity  # 코사인 유사도

app = Flask(__name__)

with open("D:/AIB/AI_Section3_Project/odf2.pkl", "rb") as fw:
    odf2 = pickle.load(fw)

with open("D:/AIB/AI_Section3_Project/place_simi.pkl", "rb") as fw:
    place_simi_co_sorted_ind = pickle.load(fw)

count_vect_category = CountVectorizer(min_df=0, ngram_range=(1,2))
place_category = count_vect_category.fit_transform(odf2['cate_mix']) 
place_simi_cate = cosine_similarity(place_category, place_category) 
place_simi_cate_sorted_ind = place_simi_cate.argsort()[:, ::-1]
place_simi_co = (
                + place_simi_cate * 0.02 # 공식 1. 카테고리 유사도
                + np.repeat([odf2['naver_blog_review_qty'].values], len(odf2['naver_blog_review_qty']) , axis=0) * 0.0003  # 공식 2. 블로그 리뷰가 얼마나 많이 올라왔는지
                + np.repeat([odf2['naver_star_point'].values], len(odf2['naver_star_point']) , axis=0) * 0.0005            # 공식 3. 블로그 별점이 얼마나 높은지
                + np.repeat([odf2['visitor_review'].values], len(odf2['visitor_review']) , axis=0) * 0.0005    # 공식 4. 방문자 리뷰가 얼마나 많이 됐는지
                + np.repeat([odf2['office_review'].values], len(odf2['office_review']) , axis=0) * 0.01 # 공식 5. 방문자 리뷰가 얼마나 많이 됐는지
                )
# 아래 place_simi_co_sorted_ind 는 그냥 바로 사용하면 됩니다.
place_simi_co_sorted_ind = place_simi_co.argsort()[:, ::-1] 
# 최종 구현 함수
def find_simi_place(odf2, sorted_ind, place_name, top_n=10):  
    place_title = odf2[odf2['name'] == place_name]
    place_index = place_title.index.values
    similar_indexes = sorted_ind[place_index, :(top_n)]
    similar_indexes = similar_indexes.reshape(-1)
    return odf2.iloc[similar_indexes]


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/Korean1/')
def Korean1():
    Korean1 = find_simi_place(odf2, place_simi_co_sorted_ind, '해운대이름난암소갈비', 20)
    Korean1 = Korean1['name'].tolist()
    return Korean1[13]

@app.route('/Korean2/')
def Korean2():
    Korean2 = find_simi_place(odf2, place_simi_co_sorted_ind, '달인막창', 20)
    Korean2 = Korean2['name'].tolist()
    return Korean2[4]
@app.route('/Korean3/')
def Korean3():
    Korean3 = find_simi_place(odf2, place_simi_co_sorted_ind, '징기스', 20)
    Korean3 = Korean3['name'].tolist()
    return Korean3[2]
@app.route('/Korean4/')
def Korean4():
    Korean4 = find_simi_place(odf2, place_simi_co_sorted_ind, '보리문디', 20)
    Korean4 = Korean4['name'].tolist()
    return Korean4[6]

@app.route('/Chinese1/')
def Chinese1():
    Chinese1 = find_simi_place(odf2, place_simi_co_sorted_ind, '홍탕', 3)
    Chinese1 = Chinese1['name'].tolist()
    return Chinese1[0]
@app.route('/Chinese2/')
def Chinese():
    return render_template("index.html")

@app.route('/Western/')
def Western():
    return render_template("index.html")

@app.route('/Snack/')
def Snack():
    return render_template("index.html")

@app.route('/Japanese/')
def Japanese():
    return render_template("index.html")

@app.route('/Cafe/')
def Cafe():
    return render_template("index.html")


