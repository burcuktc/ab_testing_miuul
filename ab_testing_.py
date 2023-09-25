#AB Testing
#Veri bilimi alanında en çok kullanılan yöntemlerden birisidir.
#AB Testi sektörde birçok yerde karşımıza çıkabilecek olan istatistiki bir metoddur. Örneğin bir eticaret sitesinde sepete ekle butonunun rengi değiştirilmek isteniyor, ancak yapılacak değişikliğin satışlara nasıl bir etkisinin olacağı bilinmiyor. Bu durumda yapılacak değişikliğin etkisi istatistiki olarak AB testi ile kanıtlabilir.
#Başka bir örnek olarak oyun zorluk seviyelerinin kolay mı yoksa zor mu yapmanız gerektiği bilgisini elde edinmek ve buna göre bölüm atlama zorlukları belirlenmek için AB testi kullanılabilir 
# #A bir özelliği ya da bir grubu temsil etsin B farklı bir özelliği ya da grubu temsil etsin. Bu ikisi arasında farklılık olup olmadığı ile ilgilendiğimiz konudur.
# Temel İstatistik Kavramları
#Sampling(Örnekleme): Örneklem, bir ana kitle içerisinden bu ana kitlenin özelliklerini iyi temsil ettiği varsayılan temsilcisidir.
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)
pd.set_option("display.float_format", lambda x: '%.5f' % x)
# Sampling (Örnekleme)
populasyon = np.random.randint(0,80,10000) # 0-80 arasında 10000 tane sayı oluşturduk. Bu sayılar varsayalım ki insanların yaşlarını temsil etsin.
populasyon.mean()

np.random.seed(115)
orneklem = np.random.choice(a=populasyon, size=100) #populasyondan 100 tane örnek seçtik
orneklem.mean()
#örneklem sayesinde daha az veriyle genellemeler yaparız. zaman, para, iş gücü bakımından avantajlar sağlar.
np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10
#örnek sayısı arttığında örneklem dağılımına ilişkin ortalama da ana kitlenin ortalamasına yakınsar.

# Descriptive Statistics (Betimsel İstatistikler)
df = sns.load_dataset("tips")
df.describe().T #count,mean,std,min,%25,..,max temel istatistik bilgileri verir.
#eğer elimizdeki değişkenin dağılımı çarpık ise yani içerisinde aykırı değelr varsa bu durumda bu değişkeni temsil etmek için ortalama değer değil medyan kullanılmalıdır.
#aykırı değer olup olmadığını medyan ve ortalama arasındaki fark çoksa aykırı değerler vardır diyebiliriz.

# Confidence Intervals (Güven Aralıkları)
#Anakütle parametresinin tahmini değerini (istatistik) kapsayabilecek iki sayıdan oluşan bir aralık bulunmasıdır.
#Örneğin web sitesinde geçirilen ortalama sürenin güven aralığı nedir?
#Ortalama : 180 saniye , Standart sapma: 40 sn
#%95 güven ile 172 saniye ile 188 saniye arasındadır denilebilir.

# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("tips")
df.describe().T
"""          count     mean     std     min      25%      50%      75%      max
total_bill 244.00000 19.78594 8.90241 3.07000 13.34750 17.79500 24.12750 50.81000
tip        244.00000  2.99828 1.38364 1.00000  2.00000  2.90000  3.56250 10.00000
size       244.00000  2.56967 0.95110 1.00000  2.00000  2.00000  3.00000  6.00000"""
#total_bill değişkeninin ortalaması 19.78 ama güven aralığı nedir öğrenmek istiyoruz. total_bill ile kötü durumda ve iyi durumda ne kazanır bunu öğrenmek isteyebiliriz örneğin.
sms.DescrStatsW(df["total_bill"]).tconfint_mean() #(18.66333170435847, 20.908553541543164)
#müşterilerin ödediği hesap ortalamaları istatistiki olarak %95 güven ile 18.66 - 20.90 aralığındadır sonucunu çıkarabiliriz.
sms.DescrStatsW(df["tip"]).tconfint_mean()
# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()


# Correlation (Korelasyon)
#Değişkenler arasındaki ilişki, bu ilişkinin yönü ve şiddeti ile ilgili bilgiler sağlayan istatiksel bir yöntemdir.
# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()
#verilen bahşişler ile ödenen hesap arasında bir korelasyon olup olmadığını merak ediyoruz.
df["total_bill"] = df["total_bill"] - df["tip"] #veri setinde yemeğin toplam fatura fiyatına bahşiş de dahildir. Bunu düzeltmek için bu adımı uyguluyoruz.

df.plot.scatter("tip", "total_bill") #scatter saçılım grafiği ile gösterelim.
plt.show()

df["tip"].corr(df["total_bill"]) #ikisi arasındaki korelasyonu değerlendirmek için # 0.5766634471096374

#Hipotez testleri
#Bir inanışı, bir savı test etmek için kullanılan istatiksel yöntemlerdir.
#Grup karşılaştırmalarında temel amaç olası farklılıkların şans eser ortaya çıkıp çıkmadığını göstermeye çalıştırmaktır.
 '''Örneğin mobil uygulamada yapılan arayüz değişikliği sonrasında kullanıcıların uygulmada geçirdiği süre arttı mı? Hipotez testleri açısından şu şekilde modellenebilir
Arayüz değişikliği öncesinde yani A grubunda, arayüz değişikliği sonrasında yani B grubunda kullanıcıların uygulmada geçirdiği süre arasında fark yoktur şeklinde bir hipotez kuruyoruz ve bunu test ediyoruz.
Tasarım 1 örneğin 55 dk, tasarım 2 58 dk. Bu farklılığa göre 2. tasarım kesinlikle daha iyidir diyemeyiz bu farklılık şans eseri ortaya çıkmış olabilir'''

# AB Testing (Bağımsız İki Örneklem T Testi)
#AB Testinde yaygınca ya iki grubun ortalaması kıyaslanıyodur ya da iki gruba ilişkin oranlar kıyaslanıyodur.
#Bağımsız iki örneklem T testi: iki grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.
#A ve B ifadeleri kontrol grubu ve deney grubu temsil etmek için kullanılır. Genellikle mobil ya da web uygulamalarında gerçekleştirilen yenilikler ya da deneme yapılması istenilen özelliklerin test edilmesi için kullanılır.
#Ya da geliştirilen bazı algoritmaların neticesinde gelirlerde ortaya farklılk çıkıp çıkmamasının test edilmesi gibi durumlarda kullanılır.
# p value değerine bakarak hipotezlerin sonucunu yorumlaycağız. hipotez testlerini gerçekleştirdiğimizde, ilgili fonksiyonları kullandığımızda bu fonksyonlar bize bir p value değeri veriyor olacak. Bu p value değeri eğer 0.05'ten küçükse  H0 red deriz ve ona göre yorumları gerçekleştiririz.
#Bağımsız iki örneklem T testinin 2 tane varsayımı vardır: 1)Normallik 2)Varyans Homojenliği
#1)Normallik:İkii grubun da normal dağılması gerekmektedir.
#2)Varyans Homojenliği:İki grubun dağılımlarının varyanslarınının birbirine benzer olup olmamasıdır.

"""1) Hiptezleri kur, 2) varsayımları incele (burada gerekirse veri önişleme, keşifçi veri analizi gibi işlemleri yap. 3) p value değrine bakarak yorum yap."""
# AB Testing (Bağımsız İki Örneklem T Testi) (1.Uygulama)
# 1. Hipotezleri Kur (örneğin iki grup arasında fark yoktur gibi)
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İstatistik Olarak Anlamlı Fark var mı?
df = sns.load_dataset("tips")
df.head()
df.groupby("smoker").agg({'total_bill': 'mean'})     #sigara içip içmeme durumuna göre ortalama aldık.  total_bill
"""smoker            
   Yes       20.75634
   No        19.18828""" #fark var gibi görünüyor ama bu fark şans eseri mi ortaya çıktı istatistiksel olarak test etmeliyiz.

# 1. Hipotezi Kur
# H0: M1 = M2 #elimizde tüm olası müşterilerin verisi olsaydı bu müşterilerin ödeyeceği hesap ortalamaları arasında fark yoktur hipotezi
# H1: M1 != M2

# 2. Varsayım Kontrolü
# Normallik Varsayımı
# Varyans Homojenliği
# Normallik Varsayımı

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"]) #shapiro bir değişkenin normal dağılım olup olmadığını test eder.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #Test Stat = 0.9367, p-value = 0.0002
# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #Test Stat = 0.9045, p-value = 0.0000
#p 0.05ten küçük çıktı. Bu durumda normal dağılım varsayımı sağlanmadı. Bu durumda non-parametric test kullanmalıyız.

# Varyans Homojenligi Varsayımı #örnekte normal dağılım olduğu çıksaydı bu varsayımı da test edecektik. Örnek olması açısından yine de kontrol edeceğiz.
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"]) #levene iki farklı gruba göre varyans homojenliği sağlanıp sağlanmadığını söyler
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 4.0537, p-value = 0.0452
#bu drumda H0 red. Varyanslar homojen değildir.
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

## 3 ve 4. Hipotezin Uygulanması
# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 1.3384, p-value = 0.1820
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 7531.5000, p-value = 0.3413
#p value değeri 0.05'ten küçük olmadığı için H0 reddedilemedi. Bu durumda sigara içenler ile içmeyenlerin hesap ortalaması araısnda istatiksel olarak fark yoktur.

# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
df = sns.load_dataset("titanic")
df.head()
df.groupby("sex").agg({'age': 'mean'})
"""    age
sex            
female 27.91571
male   30.72664"""
# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)
# 2. Varsayımları İncele
# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9848, p-value = 0.0071 kadınlar için H0 rededilir.

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9747, p-value = 0.0000 erkekler için H0 reddedilir.
# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.0013, p-value = 0.9712 #varyanslar homojendir ancak normallik sağlanmadığı için nonparametrik
test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 53212.5000, p-value = 0.0261 kadın ve erkek yolcular arasında istatiksel olarak ortalama yaş farkı vardır.


# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
df = pd.read_csv("C:/Users/asus/Desktop/miuul/diabetes.csv")
df.head()
df.groupby("Outcome").agg({"Age": "mean"})
"""      Age
Outcome         
0       31.19000
1       37.06716"""
# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue)) #Test Stat = 0.9546, p-value = 0.0000 HO reddedilir.

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Normallik varsayımı sağlanmadığı için nonparametrik.

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 92050.0000, p-value = 0.0000 H0 reddedilir yani Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark vardır.

# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?


# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("C:/Users/asus\Desktop/miuul/course_reviews.csv")
df.head()
df[(df["Progress"] > 75)]["Rating"].mean()
# kursun yüzde 75'ini izleyenlerin puan ortalaması 4.860491071428571
df[(df["Progress"] < 25)]["Rating"].mean()
#kursun yüzde 25ini izleyenlerin puan ortalaması
#4.7225029148853475
test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.3160, p-value = 0.0000
test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.5710, p-value = 0.0000 normallik varsayımı sağlanmıyor.
test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 661481.5000, p-value = 0.0000 H0 reddedilir.


######################################################
# AB Testing (İki Örneklem Oran Testi)
# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

basari_sayisi = np.array([300, 250]) #1. grubun başarı sayısı 300, 2. grubun başarı sayısı 250
gozlem_sayilari = np.array([1000, 1100]) #1. grubun gözlem sayısı 1000, 2. grubun 1100

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari) #proportions_ztest ilk argumanına başarı sayısını, 2. argumanına gözlem sayısını ister)
#Out[41]: (3.7857863233209255, 0.0001532232957772221) pvalue=0.0001532232957772221 0.05'ten küçük, H0 red, anlamlı farklılık vardır.

basari_sayisi / gozlem_sayilari # array([0.3       , 0.22727273]) 1. grubun 0.3 , 2. grubun 0.22727273

# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()
df.loc[df["sex"] == "female", 'survived'].mean() # 0.7420382165605095
df.loc[df["sex"] == "male", "survived"].mean() # 0.18890814558058924

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])  #proportions_ztest ilk argumanına başarı sayısını, 2. argumanına gözlem sayısını ister
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 16.2188, p-value = 0.0000 H0 red, Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark vardır

# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.

for group in list(df["day"].unique()): #day'in unique değerlerini bir listeye çevirip üzerinde gezilebilecek iteratif bir nesneye çevirdik.
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)
"""#Sun p-value: 0.0036
Sat p-value: 0.0000
Thur p-value: 0.0000
Fri p-value: 0.0409""" #hepsi 0.05ten küçük H0 reddedilir
# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.6654, p-value = 0.5741

# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})
"""       mean   median
day                     
Thur   17.68274 16.20000
Fri    17.15158 15.38000
Sat    20.44138 18.24000
Sun    21.41000 19.63000"""

# HO: Grup ortalamaları arasında ist ol anl fark yoktur

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])
#KruskalResult(statistic=10.403076391437086, pvalue=0.01543300820104127) #fark vardır. ama bu fark hangisinden kaynaklanıyor?
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
