# PROJE: Rating Product & Sorting Reviews in Amazon
# İş Problemi
# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
df = pd.read_csv("C:/Users/asus/Desktop/miuul/amazon_review.csv")
df.head()
"""Out[6]: 
       reviewerID        asin  reviewerName helpful                                         reviewText  overall                                 summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
0  A3SBTW3WS4IQSN  B007WTAJTO           NaN  [0, 0]                                         No issues.  4.00000                              Four Stars      1406073600  2014-07-23       138            0           0
1  A18K1ODH1I2MVB  B007WTAJTO          0mie  [0, 0]  Purchased this for my device, it worked as adv...  5.00000                           MOAR SPACE!!!      1382659200  2013-10-25       409            0           0
2  A2FII3I2MBMUIA  B007WTAJTO           1K3  [0, 0]  it works as expected. I should have sprung for...  4.00000               nothing to really say....      1356220800  2012-12-23       715            0           0
3   A3H99DFEG68SR  B007WTAJTO           1m2  [0, 0]  This think has worked out great.Had a diff. br...  5.00000  Great buy at this price!!!  *** UPDATE      1384992000  2013-11-21       382            0           0
4  A375ZM4U047O79  B007WTAJTO  2&amp;1/2Men  [0, 0]  Bought it with Retail Packaging, arrived legit...  5.00000"""
df["overall"].mean() #4.587589013224822

# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), 'overall'].mean() #günümüze en yakın paunlar #4.6957928802588995
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() # 4.64
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() # 4.57
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean() #en eski kullanıcı puanları # 4.45
# zaman bazlı ortalama ağırlıkların belirlenmesi
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22) # 4.59559316512811

df["overall"].mean() # 4.58
#günümüze yakın olan puanların etkisini daha yukarı çekecek şekilde bir puanlama sağlamış olduk.

# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
# Adım 1. helpful_no Değişkenini Üretiniz

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]] #verisetinde görmek istediğimiz kolonları getirdik
df.head()
"""Out[12]: 
   reviewerName  overall                                 summary  helpful_yes  helpful_no  total_vote  reviewTime
0           NaN  4.00000                              Four Stars            0           0           0  2014-07-23
1          0mie  5.00000                           MOAR SPACE!!!            0           0           0  2013-10-25
2           1K3  4.00000               nothing to really say....            0           0           0  2012-12-23
3           1m2  5.00000  Great buy at this price!!!  *** UPDATE            0           0           0  2013-11-21
4  2&amp;1/2Men  5.00000                        best deal around            0           0           0  2013-07-13"""
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

#####################
# score_pos_neg_diff
#####################


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_pos_neg_diff", ascending=False).head(20)



# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("score_average_rating", ascending=False).head(20)



# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values("wilson_lower_bound", ascending=False).head(20)



##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
"""
Out[13]: 
                              reviewerName  overall                                            summary  helpful_yes  helpful_no  total_vote  reviewTime  score_pos_neg_diff  score_average_rating  wilson_lower_bound
2031                  Hyoun Kim "Faluzure"  5.00000  UPDATED - Great w/ Galaxy S4 & Galaxy Tab 4 10...         1952          68        2020  2013-01-05                1884               0.96634             0.95754
3449                     NLee the Engineer  5.00000  Top of the class among all (budget-priced) mic...         1428          77        1505  2012-09-26                1351               0.94884             0.93652
4212                           SkincareCEO  1.00000  1 Star reviews - Micro SDXC card unmounts itse...         1568         126        1694  2013-05-08                1442               0.92562             0.91214
317                Amazon Customer "Kelly"  1.00000                                Warning, read this!          422          73         495  2012-02-09                 349               0.85253             0.81858
4672                               Twister  5.00000  Super high capacity!!!  Excellent price (on Am...           45           4          49  2014-07-03                  41               0.91837             0.80811
1835                           goconfigure  5.00000                                           I own it           60           8          68  2014-02-28                  52               0.88235             0.78465
3981            R. Sutton, Jr. "RWSynergy"  5.00000  Resolving confusion between "Mobile Ultra" and...          112          27         139  2012-10-22                  85               0.80576             0.73214
3807                            R. Heisler  3.00000   Good buy for the money but wait, I had an issue!           22           3          25  2013-02-27                  19               0.88000             0.70044
4306                         Stellar Eller  5.00000                                      Awesome Card!           51          14          65  2012-09-06                  37               0.78462             0.67033
4596           Tom Henriksen "Doggy Diner"  1.00000     Designed incompatibility/Don't support SanDisk           82          27         109  2012-09-22                  55               0.75229             0.66359
315             Amazon Customer "johncrea"  5.00000  Samsung Galaxy Tab2 works with this card if re...           38          10          48  2012-08-13                  28               0.79167             0.65741
1465                              D. Stein  4.00000                                           Finally.            7           0           7  2014-04-14                   7               1.00000             0.64567
1609                                Eskimo  5.00000                  Bet you wish you had one of these            7           0           7  2014-03-26                   7               1.00000             0.64567
4302                             Stayeraug  5.00000                        Perfect with GoPro Black 3+           14           2          16  2014-03-21                  12               0.87500             0.63977
4072                           sb21 "sb21"  5.00000               Used for my Samsung Galaxy Tab 2 7.0            6           0           6  2012-11-09                   6               1.00000             0.60967
1072                        Crysis Complex  5.00000               Works wonders for the Galaxy Note 2!            5           0           5  2012-05-10                   5               1.00000             0.56552
2583                               J. Wong  5.00000                  Works Great with a GoPro 3 Black!            5           0           5  2013-08-06                   5               1.00000             0.56552
121                                 A. Lee  5.00000                     ready for use on the Galaxy S3            5           0           5  2012-05-09                   5               1.00000             0.56552
1142  Daniel Pham(Danpham_X @ yahoo.  com)  5.00000                          Great large capacity card            5           0           5  2014-02-04                   5               1.00000             0.56552
1753                             G. Becker  5.00000                    Use Nothing Other Than the Best            5           0           5  2012-10-22                   5               1.00000             0.56552"""
