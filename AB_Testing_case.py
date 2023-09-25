# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
# İş Problemi
# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olanbombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için birA/B testiyapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.

# Veri Seti Hikayesi
# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç

# Görev 1:  Veriyi Hazırlama ve Analiz Etme
# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind
pd.set_option("display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

dataframe_control = pd.read_excel("C:/Users/asus/Desktop/miuul/ab_testing.xlsx" , sheet_name="Control Group")
dataframe_test = pd.read_excel("C:/Users/asus/Desktop/miuul/ab_testing.xlsx" , sheet_name="Test Group")
df_control = dataframe_control.copy()
df_test = dataframe_test.copy()

def check_df(dataframe,head=5):
    print("##################### Shape #####################")
    print(dataframe.shape) #satır-sütun bilgisi
    print("##################### Types #####################")
    print(dataframe.dtypes) #her kolonun tip bilgisi
    print("##################### Head #####################")
    print(dataframe.head()) #
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T) #nümerik kolonların verilen listedeki çeyreklerine göre incelemesi yapılır

check_df(df_control)
"""##################### Shape #####################
(40, 4)
##################### Types #####################
Impression    float64
Click         float64
Purchase      float64
Earning       float64
dtype: object
##################### Head #####################
    Impression      Click  Purchase    Earning
0  82529.45927 6090.07732 665.21125 2311.27714
1  98050.45193 3382.86179 315.08489 1742.80686
2  82696.02355 4167.96575 458.08374 1797.82745
3 109914.40040 4910.88224 487.09077 1696.22918
4 108457.76263 5987.65581 441.03405 1543.72018
##################### Tail #####################
     Impression      Click  Purchase    Earning
35 132064.21900 3747.15754 551.07241 2256.97559
36  86409.94180 4608.25621 345.04603 1781.35769
37 123678.93423 3649.07379 476.16813 2187.72122
38 101997.49410 4736.35337 474.61354 2254.56383
39 121085.88122 4285.17861 590.40602 1289.30895
##################### NA #####################
Impression    0
Click         0
Purchase      0
Earning       0
dtype: int64
##################### Quantiles #####################
               0.00000     0.05000     0.50000      0.95000      0.99000      1.00000
Impression 45475.94296 79412.01792 99790.70108 132950.53245 143105.79110 147539.33633
Click       2189.75316  3367.48426  5001.22060   7374.36120   7761.79511   7959.12507
Purchase     267.02894   328.66242   531.20631    748.27076    790.18779    801.79502
Earning     1253.98952  1329.57708  1975.16052   2318.52850   2481.30874   2497.29522 #0.99 ile 1 arasında aşırı bir artış yok demek ki aykırı değer yoktur yorumunu yapabiliriz.
"""
check_df(df_test)
"""##################### Shape #####################
(40, 4)
##################### Types #####################
Impression    float64
Click         float64
Purchase      float64
Earning       float64
dtype: object
##################### Head #####################
    Impression      Click  Purchase    Earning
0 120103.50380 3216.54796 702.16035 1939.61124
1 134775.94336 3635.08242 834.05429 2929.40582
2 107806.62079 3057.14356 422.93426 2526.24488
3 116445.27553 4650.47391 429.03353 2281.42857
4 145082.51684 5201.38772 749.86044 2781.69752
##################### Tail #####################
     Impression      Click  Purchase    Earning
35  79234.91193 6002.21358 382.04712 2277.86398
36 130702.23941 3626.32007 449.82459 2530.84133
37 116481.87337 4702.78247 472.45373 2597.91763
38  79033.83492 4495.42818 425.35910 2595.85788
39 102257.45409 4800.06832 521.31073 2967.51839
##################### NA #####################
Impression    0
Click         0
Purchase      0
Earning       0
dtype: int64
##################### Quantiles #####################
               0.00000     0.05000      0.50000      0.95000      0.99000      1.00000
Impression 79033.83492 83150.50378 119291.30077 153178.69106 158245.26380 158605.92048
Click       1836.62986  2600.36102   3931.35980   5271.18691   6012.87730   6019.69508
Purchase     311.62952   356.69540    551.35573    854.20895    876.57610    889.91046
Earning     1939.61124  2080.97621   2544.66611   2931.31145   3091.94089   3171.48971"""

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df_control["group"] = "control"
df_test["group"] = "test"
#iki farklı veriyi birleştirmeden önce ayrımlarını yapabilmemiz için iki veriyi de etiketliyoruz
df = pd.concat([df_control,df_test], axis=0,ignore_index=True) #concat ile birleştirme yapıyoruz, axis=0 ile satır bazlı yani alt alta birleştirme yapıyoruz, axis=1 olsaydı yan yana birleşecekti, ignore_index=true ile ilk veri bitip ikinci veriye geçtiğinde indexi sıfırdan değil kaldığı yerden brileştirmeye devam eder)

df.head()
"""    Impression      Click  Purchase    Earning    group
0  82529.45927 6090.07732 665.21125 2311.27714  control
1  98050.45193 3382.86179 315.08489 1742.80686  control
2  82696.02355 4167.96575 458.08374 1797.82745  control
3 109914.40040 4910.88224 487.09077 1696.22918  control
4 108457.76263 5987.65581 441.03405 1543.72018  control"""
df.tail()
"""  Impression      Click  Purchase    Earning group
75  79234.91193 6002.21358 382.04712 2277.86398  test
76 130702.23941 3626.32007 449.82459 2530.84133  test
77 116481.87337 4702.78247 472.45373 2597.91763  test
78  79033.83492 4495.42818 425.35910 2595.85788  test
79 102257.45409 4800.06832 521.31073 2967.51839  test"""

# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
# Adım 1: Hipotezi tanımlayınız.

# H0 : M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
# H1 : M1!= M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

df.groupby("group").agg({"Purchase": "mean"})
""" Purchase
group            
control 550.89406
test    582.10610"""

# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz
# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır
# p < 0.05 H0 RED
# p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ?
# Elde edilen p-valuedeğerlerini yorumlayınız.


test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.5891
# HO reddedilemez. Control grubunun değerleri normal dağılım varsayımını sağlamaktadır.

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value=0.1541
# HO reddedilemez. test grubunun değerleri normal dağılım varsayımını sağlamaktadır.

# Varyans Homojenliği :
#varyans, verilerin aritmetik ortalamadan sapmalarının karelerinin ortalamasıdır. yani standart sapmanın karekök alınmamış halidir.
# H0: Varyanslarhomojendir.
# H1: Varyanslarhomojen Değildir.
# p < 0.05 H0 RED
# p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu? Elde edilen p-valuedeğerlerini yorumlayınız.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value=0.1083
# HO reddedilemez. Control ve Test grubunun değerleri varyans homejenliği varsayımını sağlamaktadır.
# Varyanslar Homojendir.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

# Varsayımlar sağlandığı için bağımsız iki örneklem t testi (parametrik test) yapılmaktadır.
# H0: M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında ist. ol.anl.fark yoktur.)
# H1: M1 != M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında ist. ol.anl.fark vardır)
# p<0.05 HO RED , p>0.05 HO REDDEDİLEMEZ

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.3493
# Adım 3: Test sonucunda elde edilen p_valuedeğerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# p-value=0.3493
# HO reddedilemez. Kontrol ve test grubu satın alma ortalamaları arasında istatistiksel olarak anlamlı farklılık yoktur.


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
#İlk önce iki gruba da normallik testi uygulanmıştır. İki grubun da normal dağılıma uygun olduğu gözlemlendiğinden
#ikinci varsayıma geçilerek varyansın homojenliği incelenmiştir. Varyanslar homojen çıktığından
#Bağımsız 2 örneklem T testi uygulanmıştır. Uygulama sonucunda p-değerinin 0.05'ten büyük olduğu gözlemlenmiştir ve H0 reddedilememiştir.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
#Satın alma anlamında anlamlı bir fark olmadığından müşteri iki yöntemden birini seçebilir fakat burada diğer istatistiklerdeki farklar
#da önem arz edecektir. Tıklanma, Etkileşim, kazanç  ve dönüşüm oranlarındaki farklılıklar değerlendirilip hangi
#yöntemin daha kazançlı olduğu tespit edilebilir. Özellikle facebook'ta tıklama başına para ödendiği için hangi yöntemde tıklanma oranının daha düşük
#olduğu tespit edilip ve CTR(Click through-rate - tıklama oranı) bakılabilir.