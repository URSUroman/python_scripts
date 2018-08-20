
#from third-party
import numpy as np
import scipy as sp
from scipy.io import loadmat
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import glob
from threading import Thread



##Check location of libraries
#npdir = os.path.dirname(np.__file__)
#print("NumPy is installed in %s" % npdir)
#
#spdir = os.path.dirname(sp.__file__)
#print("SciPy is installed in %s" % spdir)
#
#
#print(np.__version__)
#print(sp.__version__)
##End check location


pwd = os.getcwd()

os.chdir("..\SyllablesClassification\Koumura Data Set\Song_Data\Test_Songs_predict")
#retval = os.getcwd()
#print("Current working directory %s" % retval)	

#Take all files from the directory
songfiles_list = glob.glob('*.txt')	
#print("songfiles_list: %s" % songfiles_list)
os.chdir("..")

count_k_annot=0
count_n_annot=0
count_i_annot=0
count_a_annot=0
count_b_annot=0
count_c_annot=0
count_d_annot=0
count_e_annot=0

count_k_pred=0
count_n_pred=0
count_i_pred=0
count_a_pred=0
count_b_pred=0
count_c_pred=0
count_d_pred=0
count_e_pred=0

#k, n, i, a, b, c, d, e
stats_k = np.zeros((8), dtype=np.int)
stats_n = np.zeros((8), dtype=np.int)
stats_i = np.zeros((8), dtype=np.int)
stats_a = np.zeros((8), dtype=np.int)
stats_b = np.zeros((8), dtype=np.int)
stats_c = np.zeros((8), dtype=np.int)
stats_d = np.zeros((8), dtype=np.int)
stats_e = np.zeros((8), dtype=np.int)




#file_num is the index of the file in the songfiles_list
#For each annotation see if the prediction is correct
for file_num, songfile in enumerate(songfiles_list):	

    # Write file with onsets, offsets, labels
    current_dir = os.getcwd()
    file_path_predict = current_dir+'\Test_Songs_predict\\'+songfile
    file_path_annot = current_dir+'\Test_Songs_annot\\'+songfile[0:15]+'_annot.txt'
	
    with open(file_path_predict) as fp:
	
        with open(file_path_annot) as fa:
	
             lines_p = fp.readlines()
             lines_a = fa.readlines()

             for line_p,line_a in zip(lines_p,lines_a):
                 #line = str(lines)
                 splt_p = line_p.split(",")
                 splt_a = line_a.split(",")
                 #onsets.append(float(splt[0]))
                 #offsets.append(float(splt[1]))
                 #labels.append(float(splt[2]))
		         
                 splt_p_bis = (splt_p[2]).split("\n")
                 splt_a_bis = (splt_a[2]).split("\n")
                 #print("syl: %s" % splt_a_bis[0])
                 #print("splt_a_bis: %s" % splt_a_bis)
                 if splt_a_bis[0]=='k':
                    count_k_annot+=1
                    if splt_p_bis[0] == 'k':
                       count_k_pred+=1
                       stats_k[0]+=1
                    elif splt_p_bis[0] == 'n':
                       stats_k[1]+=1
                    elif splt_p_bis[0] == 'i':
                       stats_k[2]+=1
                    elif splt_p_bis[0] == 'a':
                       stats_k[3]+=1
                    elif splt_p_bis[0] == 'b':
                       stats_k[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       stats_k[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       stats_k[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       stats_k[7]+=1					   
					   
                 elif splt_a_bis[0]=='n':
                    count_n_annot+=1
                    if splt_p_bis[0] == 'k':
                       stats_n[0]+=1
                    elif splt_p_bis[0] == 'n':
                       count_n_pred+=1
                       stats_n[1]+=1
                    elif splt_p_bis[0] == 'i':
                       stats_n[2]+=1
                    elif splt_p_bis[0] == 'a':
                       stats_n[3]+=1
                    elif splt_p_bis[0] == 'b':
                       stats_n[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       stats_n[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       stats_n[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       stats_n[7]+=1	
					   
                 elif splt_a_bis[0]=='i':
                    count_i_annot+=1
                    if splt_p_bis[0] == 'k':
                       stats_i[0]+=1
                    elif splt_p_bis[0] == 'n':
                       stats_i[1]+=1
                    elif splt_p_bis[0] == 'i':
                       count_i_pred+=1
                       stats_i[2]+=1
                    elif splt_p_bis[0] == 'a':
                       stats_i[3]+=1
                    elif splt_p_bis[0] == 'b':
                       stats_i[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       stats_i[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       stats_i[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       stats_i[7]+=1	
 
                 elif splt_a_bis[0]=='a':
                    count_a_annot+=1
                    if splt_p_bis[0] == 'k':
                       stats_a[0]+=1
                    elif splt_p_bis[0] == 'n':
                       stats_a[1]+=1
                    elif splt_p_bis[0] == 'i':
                       stats_a[2]+=1
                    elif splt_p_bis[0] == 'a':
                       count_a_pred+=1
                       stats_a[3]+=1
                    elif splt_p_bis[0] == 'b':
                       stats_a[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       stats_a[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       stats_a[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       stats_a[7]+=1
					   
                 elif splt_a_bis[0]=='b':
                    count_b_annot+=1
                    if splt_p_bis[0] == 'k':
                       stats_b[0]+=1
                    elif splt_p_bis[0] == 'n':
                       stats_b[1]+=1
                    elif splt_p_bis[0] == 'i':
                       stats_b[2]+=1
                    elif splt_p_bis[0] == 'a':
                       stats_b[3]+=1
                    elif splt_p_bis[0] == 'b':
                       count_b_pred+=1
                       stats_b[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       stats_b[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       stats_b[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       stats_b[7]+=1
					   
                 elif splt_a_bis[0]=='c':
                    count_c_annot+=1
                    if splt_p_bis[0] == 'k':
                       stats_c[0]+=1
                    elif splt_p_bis[0] == 'n':
                       stats_c[1]+=1
                    elif splt_p_bis[0] == 'i':
                       stats_c[2]+=1
                    elif splt_p_bis[0] == 'a':
                       stats_c[3]+=1
                    elif splt_p_bis[0] == 'b':
                       stats_c[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       count_c_pred+=1
                       stats_c[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       stats_c[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       stats_c[7]+=1

                 elif splt_a_bis[0]=='d':
                    count_d_annot+=1
                    if splt_p_bis[0] == 'k':
                       stats_d[0]+=1
                    elif splt_p_bis[0] == 'n':
                       stats_d[1]+=1
                    elif splt_p_bis[0] == 'i':
                       stats_d[2]+=1
                    elif splt_p_bis[0] == 'a':
                       stats_d[3]+=1
                    elif splt_p_bis[0] == 'b':
                       stats_d[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       stats_d[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       count_d_pred+=1
                       stats_d[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       stats_d[7]+=1

                 elif splt_a_bis[0]=='e':
                    count_e_annot+=1
                    if splt_p_bis[0] == 'k':
                       stats_e[0]+=1
                    elif splt_p_bis[0] == 'n':
                       stats_e[1]+=1
                    elif splt_p_bis[0] == 'i':
                       stats_e[2]+=1
                    elif splt_p_bis[0] == 'a':
                       stats_e[3]+=1
                    elif splt_p_bis[0] == 'b':
                       stats_e[4]+=1					   
                    elif splt_p_bis[0] == 'c':
                       stats_e[5]+=1					   
                    elif splt_p_bis[0] == 'd':
                       stats_e[6]+=1					   
                    elif splt_p_bis[0] == 'e':
                       count_e_pred+=1
                       stats_e[7]+=1

        fa.close	
    fp.close

	

performance_k=(count_k_pred/count_k_annot)*100
performance_i=(count_i_pred/count_i_annot)*100
performance_n=(count_n_pred/count_n_annot)*100
performance_a=(count_a_pred/count_a_annot)*100
performance_b=(count_b_pred/count_b_annot)*100
performance_c=(count_c_pred/count_c_annot)*100
performance_d=(count_d_pred/count_d_annot)*100
performance_e=(count_e_pred/count_e_annot)*100
performance_syl=((count_a_pred+count_b_pred+count_c_pred+count_d_pred+count_e_pred)/(count_a_annot+count_b_annot+count_c_annot+count_d_annot+count_e_annot))*100


print("***********************************************************************")
print("*                    Performances                                     *")
print("***********************************************************************")

print("nb annot k: %f" % count_k_annot)
print("nb correct pred k: %f" % count_k_pred)
print("perfomance k: %f" % performance_k)
print("stats k: %d %d %d %d %d %d %d %d\n" % (stats_k[0],stats_k[1],stats_k[2],stats_k[3],stats_k[4],stats_k[5],stats_k[6],stats_k[7]))

print("nb annot i: %f" % count_i_annot)
print("nb correct pred i: %f" % count_i_pred)
print("perfomance i: %f" % performance_i)
print("stats i: %d %d %d %d %d %d %d %d\n" % (stats_i[0],stats_i[1],stats_i[2],stats_i[3],stats_i[4],stats_i[5],stats_i[6],stats_i[7]))

print("nb annot n: %f" % count_n_annot)
print("nb correct pred n: %f" % count_n_pred)
print("perfomance n: %f" % performance_n)
print("stats n: %d %d %d %d %d %d %d %d\n" % (stats_n[0],stats_n[1],stats_n[2],stats_n[3],stats_n[4],stats_n[5],stats_n[6],stats_n[7]))

print("nb annot a: %f" % count_a_annot)
print("nb correct pred a: %f" % count_a_pred)
print("perfomance a: %f" % performance_a)
print("stats a: %d %d %d %d %d %d %d %d\n" % (stats_a[0],stats_a[1],stats_a[2],stats_a[3],stats_a[4],stats_a[5],stats_a[6],stats_a[7]))

print("nb annot b: %f" % count_b_annot)
print("nb correct pred b: %f" % count_b_pred)
print("perfomance b: %f" % performance_b)
print("stats b: %d %d %d %d %d %d %d %d\n" % (stats_b[0],stats_b[1],stats_b[2],stats_b[3],stats_b[4],stats_b[5],stats_b[6],stats_b[7]))

print("nb annot c: %f" % count_c_annot)
print("nb correct pred c: %f" % count_c_pred)
print("perfomance c: %f" % performance_c)
print("stats c: %d %d %d %d %d %d %d %d\n" % (stats_c[0],stats_c[1],stats_c[2],stats_c[3],stats_c[4],stats_c[5],stats_c[6],stats_c[7]))

print("nb annot d: %f" % count_d_annot)
print("nb correct pred d: %f" % count_d_pred)
print("perfomance d: %f" % performance_d)
print("stats d: %d %d %d %d %d %d %d %d\n" % (stats_d[0],stats_d[1],stats_d[2],stats_d[3],stats_d[4],stats_d[5],stats_d[6],stats_d[7]))

print("nb annot e: %f" % count_e_annot)
print("nb correct pred e: %f" % count_e_pred)
print("perfomance e: %f" % performance_e)
print("stats e: %d %d %d %d %d %d %d %d\n" % (stats_e[0],stats_e[1],stats_e[2],stats_e[3],stats_e[4],stats_e[5],stats_e[6],stats_e[7]))

print("nb annot syl: %f" % (count_a_annot+count_b_annot+count_c_annot+count_d_annot+count_e_annot))
print("nb correct pred syl: %f" % (count_a_pred+count_b_pred+count_c_pred+count_d_pred+count_e_pred))
print("perfomance syl: %f\n" % performance_syl)




print("***********************************************************************")
print("*            Recapitulatif des statistiques                           *")
print("***********************************************************************")


print("k n i a b c d e")
print(stats_k)
#print("\n")
print(stats_n)
#print("\n")
print(stats_i)
#print("\n")
print(stats_a)
#print("\n")
print(stats_b)
#print("\n")
print(stats_c)
#print("\n")
print(stats_d)
#print("\n")
print(stats_e)



###Redo the loop
##count_k_annot=0
##count_n_annot=0
##count_i_annot=0
##count_a_annot=0
##count_b_annot=0
##count_c_annot=0
##count_d_annot=0
##count_e_annot=0
##
##count_k_pred=0
##count_n_pred=0
##count_i_pred=0
##count_a_pred=0
##count_b_pred=0
##count_c_pred=0
##count_d_pred=0
##count_e_pred=0
##
###file_num is the index of the file in the songfiles_list
###For each prediction see if the annotation was correct
##for file_num, songfile in enumerate(songfiles_list):	
##
##    # Write file with onsets, offsets, labels
##    current_dir = os.getcwd()
##    file_path_predict = current_dir+'\Test_Songs_predict\\'+songfile
##    file_path_annot = current_dir+'\Test_Songs_annot\\'+songfile[0:15]+'_annot.txt'
##	
##    with open(file_path_predict) as fp:
##	
##        with open(file_path_annot) as fa:
##	
##             lines_p = fp.readlines()
##             lines_a = fa.readlines()
##
##             for line_p,line_a in zip(lines_p,lines_a):
##                 #line = str(lines)
##                 splt_p = line_p.split(",")
##                 splt_a = line_a.split(",")
##                 #onsets.append(float(splt[0]))
##                 #offsets.append(float(splt[1]))
##                 #labels.append(float(splt[2]))
##		         
##                 splt_p_bis = (splt_p[2]).split("\n")
##                 splt_a_bis = (splt_a[2]).split("\n")
##                 #print("syl: %s" % splt_a_bis[0])
##                 #print("splt_a_bis: %s" % splt_a_bis)
##                 if splt_p_bis[0]=='k':
##                    count_k_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_k_annot+=1
##                 elif splt_p_bis[0]=='n':
##                    count_n_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_n_annot+=1
##                 elif splt_p_bis[0]=='i':
##                    count_i_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_i_annot+=1
##                 elif splt_p_bis[0]=='a':
##                    count_a_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_a_annot+=1
##                 elif splt_p_bis[0]=='b':
##                    count_b_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_b_annot+=1
##                 elif splt_p_bis[0]=='c':
##                    count_c_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_c_annot+=1
##                 elif splt_p_bis[0]=='d':
##                    count_d_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_d_annot+=1
##                 elif splt_p_bis[0]=='e':
##                    count_e_pred+=1
##                    if splt_a_bis[0] == splt_p_bis[0]:
##                       count_e_annot+=1
##
##        fa.close	
##    fp.close
##
##	
##
##performance_k=(count_k_annot/count_k_pred)*100
##performance_i=(count_i_annot/count_i_pred)*100
##performance_n=(count_n_annot/count_n_pred)*100
##performance_a=(count_a_annot/count_a_pred)*100
##performance_b=(count_b_annot/count_b_pred)*100
##performance_c=(count_c_annot/count_c_pred)*100
##performance_d=(count_d_annot/count_d_pred)*100
##performance_e=(count_e_annot/count_e_pred)*100
##performance_syl=((count_a_annot+count_b_annot+count_c_annot+count_d_annot+count_e_annot)/(count_a_pred+count_b_pred+count_c_pred+count_d_pred+count_e_pred))*100
##
##print("Other way round \n")
##
##print("nb pred k: %f" % count_k_pred)
##print("nb correct annot k: %f" % count_k_annot)
##print("perfomance k: %f \n" % performance_k)
##
##print("nb pred i: %f" % count_i_pred)
##print("nb correct annot i: %f" % count_i_annot)
##print("perfomance i: %f \n" % performance_i)
##
##print("nb pred n: %f" % count_n_pred)
##print("nb correct annot n: %f" % count_n_annot)
##print("perfomance n: %f \n" % performance_n)
##
##print("nb pred a: %f" % count_a_pred)
##print("nb correct annot a: %f" % count_a_annot)
##print("perfomance a: %f \n" % performance_a)
##
##print("nb pred b: %f" % count_b_pred)
##print("nb correct annot b: %f" % count_b_annot)
##print("perfomance b: %f \n" % performance_b)
##
##print("nb pred c: %f" % count_c_pred)
##print("nb correct annot c: %f" % count_c_annot)
##print("perfomance c: %f \n" % performance_c)
##
##print("nb pred d: %f" % count_d_pred)
##print("nb correct annot d: %f" % count_d_annot)
##print("perfomance d: %f \n" % performance_d)
##
##print("nb pred e: %f" % count_e_pred)
##print("nb correct annot e: %f" % count_e_annot)
##print("perfomance e: %f \n" % performance_e)
##
##print("nb pred syl: %f" % (count_a_pred+count_b_pred+count_c_pred+count_d_pred+count_e_pred))
##print("nb correct annot syl: %f" % (count_a_annot+count_b_annot+count_c_annot+count_d_annot+count_e_annot))
##print("perfomance syl: %f \n" % performance_syl)
##
##


