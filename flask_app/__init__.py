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
                + place_simi_cate * 0.7 # 공식 1. 카테고리 유사도
                + np.repeat([odf2['naver_blog_review_qty'].values], len(odf2['naver_blog_review_qty']) , axis=0) * 0.001  # 공식 2. 블로그 리뷰가 얼마나 많이 올라왔는지
                + np.repeat([odf2['naver_star_point'].values], len(odf2['naver_star_point']) , axis=0) * 0.005            # 공식 3. 블로그 별점이 얼마나 높은지
                + np.repeat([odf2['visitor_review'].values], len(odf2['visitor_review']) , axis=0) * 0.001    # 공식 4. 방문자 리뷰가 얼마나 많이 됐는지
                + np.repeat([odf2['office_review'].values], len(odf2['office_review']) , axis=0) * 0.005 # 공식 5. 업무추진비 리뷰가 얼마나 많이 됐는지
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
    return Korean1[5]

@app.route('/Korean2/')
def Korean2():
    Korean2 = find_simi_place(odf2, place_simi_co_sorted_ind, '달인막창', 20)
    Korean2 = Korean2['name'].tolist()
    return Korean2[1]
@app.route('/Korean3/')
def Korean3():
    Korean3 = find_simi_place(odf2, place_simi_co_sorted_ind, '징기스', 20)
    Korean3 = Korean3['name'].tolist()
    return Korean3[1]
@app.route('/Korean4/')
def Korean4():
    Korean4 = find_simi_place(odf2, place_simi_co_sorted_ind, '보리문디', 20)
    Korean4 = Korean4['name'].tolist()
    return Korean4[2]

@app.route('/Chinese1/')
def Chinese1():
    Chinese1 = find_simi_place(odf2, place_simi_co_sorted_ind, '홍탕', 3)
    Chinese1 = Chinese1['name'].tolist()
    return Chinese1[1]
@app.route('/Chinese2/')
def Chinese2():
    Chinese2 = find_simi_place(odf2, place_simi_co_sorted_ind, '하오', 3)
    Chinese2 = Chinese2['name'].tolist()
    return Chinese2[0]

@app.route('/Western1/')
def Western1():
    Western1 = find_simi_place(odf2, place_simi_co_sorted_ind, '오페라하우스', 3)
    Western1 = Western1['name'].tolist()
    return Western1[0]
@app.route('/Western2/')
def Western2():
    Western2 = find_simi_place(odf2, place_simi_co_sorted_ind, '콩부인더오븐', 3)
    Western2 = Western2['name'].tolist()
    return Western2[1]
@app.route('/Western3/')
def Western3():
    Western3 = find_simi_place(odf2, place_simi_co_sorted_ind, 'ARTISTA', 3)
    Western3 = Western3['name'].tolist()
    return Western3[2]

@app.route('/Snack1/')
def Snack1():
    Snack1 = find_simi_place(odf2, place_simi_co_sorted_ind, '소문난칼국수', 3)
    Snack1 = Snack1['name'].tolist()
    return Snack1[0]
@app.route('/Snack2/')
def Snack2():
    Snack2 = find_simi_place(odf2, place_simi_co_sorted_ind, '부산동창오뎅', 3)
    Snack2 = Snack2['name'].tolist()
    return Snack2[0]

@app.route('/Japanese1/')
def Japanese1():
    Japanese1 = find_simi_place(odf2, place_simi_co_sorted_ind, '하루참치', 3)
    Japanese1 = Japanese1['name'].tolist()
    return Japanese1[1]
@app.route('/Japanese2/')
def Japanese2():
    Japanese2 = find_simi_place(odf2, place_simi_co_sorted_ind, '할매집원조복국', 5)
    Japanese2 = Japanese2['name'].tolist()
    return Japanese2[3]
@app.route('/Japanese3/')
def Japanese3():
    Japanese3 = find_simi_place(odf2, place_simi_co_sorted_ind, '이노시시', 3)
    Japanese3 = Japanese3['name'].tolist()
    return Japanese3[0]

@app.route('/Cafe1/')
def Cafe1():
    Cafe1 = find_simi_place(odf2, place_simi_co_sorted_ind, '안녕커피sea', 3)
    Cafe1 = Cafe1['name'].tolist()
    return Cafe1[0]
@app.route('/Cafe2/')
def Cafe2():
    Cafe2 = find_simi_place(odf2, place_simi_co_sorted_ind, '커피나무', 3)
    Cafe2 = Cafe2['name'].tolist()
    return Cafe2[1]
@app.route('/Cafe3/')
def Cafe3():
    Cafe3 = find_simi_place(odf2, place_simi_co_sorted_ind, '테라스카페', 3)
    Cafe3 = Cafe3['name'].tolist()
    return Cafe3[2]

