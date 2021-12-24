from scipy import misc
import ModulePictureDrawFonction as fct
import matplotlib.image as mpimg
import urllib.request


# Image de départ
face = misc.face()
faceGrey = misc.face(gray=True)

print("Saisissez le numéro de l'exo : \n(0, 1, 2, 3, 4, 5, exit)")
choice = input()
while choice != "exit":
    if choice == "0":
        face = misc.face()
        fct.drawPicture(face)
        fct.drawPicture(faceGrey)
    elif choice == "1":
        # Exo 1.1
        face = misc.face(gray=True)
        miFace = fct.reducePicture(face, 4, 4, 0)
        fct.drawPicture(miFace)
    elif choice == "2":
        # Exo 1.2
        miFace = fct.reducePicture(face, 4, 4, 0)
        miFaceClean = fct.cleanColor(miFace, 150, 255)
        fct.drawPicture(miFaceClean)
    elif choice == "3":
        # Exo 1.3
        # Écrire un programme permettant d’afficher la moitié inférieure de l’image img avec une inversion du contraste.
        miFaceBottom = fct.reducePicture(face, 4, 1, 1)
        contrast = 255 - miFaceBottom
        fct.drawPicture(contrast)
    elif choice == "4":
        # Exo 1.4
        # Écrire un programme permettant d’utiliser trois niveaux de gris uniquement pour représenter l’image img :
        # les niveaux de gris entre 0 et 80 inclus sont rem-placés par 60,
        # les niveaux de gris entre 80 et 150 inclus sont remplacés par 120 et
        # les autres niveaux de gris sont remplacés par 220.
        faceChangeGrey = fct.limitGray(face.copy())
        fct.drawPicture(faceChangeGrey)
    elif choice == "5":
        # Exo 1.5
        # Écrire un programme permettant de créer une nouvelle image img2 comprenant
        # un pixel sur trois de l’image img pour la largeur et la hauteur.
        compressPic = fct.compressPict(face, 3)
        fct.drawPicture(compressPic)
    elif choice == "6":
        imgURL = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAATkAAAChCAMAAACLfThZAAABIFBMVEX////g4ODf39/e3t7z8/Pp6enw8PD5+fnt7e329vbk5OTn5+f4+Pjr6+v8/Pz/yzz/1E02bJo2dao2b5//0Ug2eK7/zD5FaIb/00v/11M2caM9Y4I2apZQcY3/yDY3Xn+YpbJ1kKv9+fDEzteKnKxedYr/88+rusjS1dmos76HoLno3LgpbaQcZZx6kaTM1d5agKHhyobXvnbh3dTk3s3/+Ofz3J5LfauYsMrxxTX41Gl5j6JWhrDvw0r52oKKnrBEf7D/3Iv/2GVhfZbHz9a6vcHc5u5qkbWivNGet856osPW4em2ydnvymf27tfk06j/xR/125PjzJRhkLrz6932zWLa1MX13Zzpy3uJq8n46Ln/3Xv/7cHjvlnq2KckUnWfbpZfAAASYElEQVR4nO2dCVfbxhbHtVmWZEukCbEDjkABE8vEMW2zkBcTaLExIaYkadI0S1/f9/8Wb5Y7kmYsyfKKMZqck8O1/paln2a5d2Y0I0k0OaqMk2KDrSvEVstgWzI9DmYR5AbYBsiLYMuQwCwLcluQU1O2wDRBrgtyB2yVl2sg18AugFzi5EqJyRVeXuLljERBIGHyJNit5eRycjm5nFxOLieXlZwqkCPfVzKTo3KBnCqS41EoAjmVkSuky4GcmpWcIB9FThXIwWOWHJrKFk0G2DrYJtglsBPkhiC3eLkJpi7Iy7y8BKYmyG2wi7y8kE1eTJBrYBf4a2VyWyDB5IyEpNIUPm6agscNdvD8eLkGdvD84uUlQR48P17ugBk+bprK6XJWTgywg4xPzbCc0GQkyMEMywlPwgabZXxWIyVVMQkFJWOdIQk5H+QBaKgzxILCV7aWPKKyDcgJdbPM10ijahihsi2PqGyXkBwqKJLZHwwGB5cNnHZ3D/rFnNwocpalNE9evfp2dHh4eA+nn+6hv472X114PaeUk0siJ6uN8/PDzVartXlv8x6kn2h6ePjoxLdycrHkym8PN4N0756ADqXDfXV5yCk0heSIGWlbqR2g4OUaHA7JxckjbSsvtzh5cb8VBy5C7uHDR76shIWbfjtsW6kdkKNnD8nRw4YgD8jxt1YWSNhgB+RsmgxTw8nUwdbBNsDWwM4qp6aWUW6Sg9+ebo4gh9idK4Ztpp9dizl7RK4nyIWziXKBBHv4Q76R6ErJvCs1yjdS4n2jIIYQfCPsdakNCg5Xcinknu6bST7omE5lkPHj5aJTmeCDLkHcqtI67vDtweWnVkI9h8j99LCx5HHroslZuySrneMLNC+S89zDhz/n5HhyDUzubp/Y/fN7SXkOpbdWTi5K7hUmdw4fvE0AR8jtyzm5CLnSEW4d3t1EcrxvpIu+Ee/XjfSNiB1pW3m56Bsh08LkNj/BBxetVHL026IPGrat4zmV8fKwbaW2ycsVyYBEnRRNB1NPsEcc1vmzjSEv0qZ1QK6Oq+eGyD0a71rHvLWs8sliCDUxhpj8cVNyR31HcvrfWmml9RF8O2MMEdYw6eVE5UOOkTGEUGdcX9xKybUOLy7fYn8uA7k84o+Si4khcnIZyaV0leTkspBLayFyconk7t2EPJfQ77D4sa/zoyPchY7TEUukRz2lbaXpmsa+yjSZJZqMIjGLOtga2AWwi/FyA2yzzMkLINfgsJ4gx9JdlA5o2mXp4PJpnCcMZ7fhbOLZbbDh1solXi7eWoI8iQSTM4DgGylJMYQ1nm8kx/tGSTEELgh4pJqdXYbBYakfS06oYbJ1TCtJTqUgZyQSYwiexHXFrUVT03Vd8yGBe94Hs68NEsgp+LBKemcdyRQq25WP+C3/8vLT4eZdmrhmFbcP4tgXR+75myC9vLJvV1+JpTbetVp3gyT4JCP6Sp5X1iprKG1tbX3Y+v35rSLnf4tgi0E3ilyFkKP0/nh2e8hZvfP7qeDGIIfRqddJjveNRox9MXlG32jIlfLf3b2fFd0wOUUVyG394o/vVE479gUEA08YHkjgCUNei/oPKCU+Pz5rOiAXYghZu2jdF8iNkeeQJ3wlkKs8xs8DflwoJ0HGhx8PHh/IhXJi8fIwhhBIgF6cLpk5+kqf/zgUfcGo1QEGl57nxiO3dSzdiri1dHR/1uROr24Fud7dYXLTldatrfe3gZz17v7oPJfeyzRE7sNH5xaQU49iyE2b5zY+Xw+5cSfUT9NCWLu/Zchz6eReDpHbejIbckpmcrRXIsxztJtCDr0SajM9mAE5sLV4uQNm4BsR+2JKcvuSdLomkvvwHbkZ9NdCr4S7tTIcDr0SXi7zcpGEDTYDzdKMPWFhFhjvCe/fnaq0Hg4U9c0Qua2vqz8LzPkWRy57C/GzpDxZHy6tb56tfNyqv5suzx1I6h9rw+TWV59c/9P9Keq5p68k6T3OciK5D89XntzBb5OTe/h035F2SJa7heQuW3HkMpbWC1MyP1YqceT+FD2pRZHjZ7fMcQQnnlwKuvBlkn0MB8CJ5La+sotZ0AhOMGpYIMkAW6dmQQMbzAKYWmEiuY7+dtLz3OHR4c/xab+hl4tlg4EbInfMLsagP8bGJMsJt6aDbYM8IwmQFxY/Ui03EvNcq3W+f4AcTpZ52GuxzA23lGfP/1pLJMeuVfBBV+YtTSuRXOudR1/sYvemCW77s/e/n1YqyeRWPOJPIIfAnfcxty9ffv31F0i/QqLW19frlUpObijPtR5JjvLs39PTjTsPSFqPJEZrPSc3RK71Cd2b//pOJA3xiyC8leQu4zzhzcO+JF2d3tnY2Lgp5MCZWdwMa7kXF309vWTgJiX34fGCZ1jrC0+DOHLnqlY6frAxDbnPi70NBnDWb5JQM34W2H5riFzLs9QnFNyk5MJepgW9SRKQo6V6EbMjXg3nucOG5R/fmRG5lY34pcvh3vRz33q2vjEVuY+3YEbOYGjsq7Uvy0/uTEXuw/tbQK74aZicNS25v77Iq0+OVXQzJfdRuRmzwKZbAWEgzo5A5JQpyb2XFz32Faw1QVO41gRNsNZEAUyzwMv1ieTaELlXRfPxVOT+0IKzFzT4NZutuhF/a2PKDUG+kJVexO5Y621LJCdJ05F7KS18pRcotQudyySXfpsxuTe4LK3+jBxZMQ9asyX3dxTFCpNTzfLF/bszJPeaQ7HK5DTJuYjG+1ORQ77czi0iJ5mX509bGcjF9w2H5La21r5+5lGs/Fua9uW3zaeQkshV/pOWcKP6n1++sIuZ8C1Nh7/WzGNfwlqwSaviWvzSsUmr4pbj5Wa8XIUVLExftaxYT/j7Dk7PIO1A4mxHSloV10lfRHfaVXEZQDH6UvmcD+sPT7cS89C7hsF8Rjkh4n9M5XJclSGukbPolZiTyM0zbo1dRS2O3Pr3Fzh9fk7TC5p24sndgr6SMcjd+YHTaz79LdTlObmY0hrXrG79npMbTS7On1vLyYVydRJy4lr9Y5LL3EKMJLe4sa/YVdRG9c+RggrkhEHfa1pF7fo8YYnvVEyPviobH1++fE1jh2P49qj1ShT+YmbtCct8nXF9a+SkkqvgdwmlF8drUXL56kIZyK2/IPoXGzm5Mcn9gN//ey0nNxa5ykf4wj85uWFyn1PG+Cssz73EFd3X5SA3agRHGfYfhodkJl0SKbpWv3aaVs/RvssdPE34wRP4triW1LgjOMp4IzgCCSnogimSFPTYUDPosSlTu5gg1wR5kZfbINcFedDBg4zy95S5TJUfuInY+R17Jf+14GxhVxc9W9AzBmeH/qASyE1Bbghyi5eHfVe8nJFgAOc7Ui3KY9bqVxX/OM2fO/1+9Q/x5zau8rX6xSrm+Y/UGGJtDc+w3nqc1D93K+NWqJy/kFmbaXFr5fTfYk4upll78bFyJ43c1l9XjpSTi3UIro4fJJGrVN5834mgyMlx5CTn6uvpxvqDB0Lbur7+5r9/sotZMnLpvUzzXWeTn+ez8/jr8Y+NO0FWW39w+uPrn3KSfOrF22N90OReJkYi46r6wkL2SXJtNnK78OzJk39fQnr/5HkhTZ7xWmcsX/ybJMpwL6HwUq4qFJRwARJyXBXliTEElWd+k4TKb95a/VJMZSvHVjFLssfcMpMzfD8nNwm5QdOtlnJyE5Dztmt7ObmJyNWqy0xuidtWRK6rZpBfU9s6W39uzH2MYlwoMzhsnyByegb375r8uYTnNzRSzTznbM9v4hii2O/pkPHVeq3WJYpCopzLyQsaqeZJLEvcap64//OhslUCcmIVs2Rx61KQ07er2zm5mZDj5Dm5zOSqezm5eZATJv6u5FuajoW/Hy6ritpW8km4YqyskMEsjZArEkvlyXFDZbJlhdP1+bEvMhddXsDYV4ntksRGGdk+RjTBTkPheCvbx4iTW6K8yEZEyS5G6lm93ul06gewS5KlSeZuHR2HTZSs3TMyBFvwe7XqdtNv9rBVxjGEUSrYeM6/XJCcYAMoRyp5HZw8n+6SBOOtBbnn1TvtTt0rOE6xWGL7RbHxVrapEk2wARQbb2XyJBJMHoy3KrxvNPMxfln1665bw6nqtn2ZeHp6o+tWJeYbyT3XdXeR6blutVpF/7ln+Msk+lJ7Xdftdrtu3WBypdjsVsMzshrG8T34IfS5p838LU2BxLzjVktFcTuKBfb29vDddnxa351sVwk5+uM9VEYROcerkoTQBeR63VoNccWgOiWQ+yfolC46I+bU7smUnH1SRcT20D+Mv1Yv3+yIX/E72zW30yBGAwHoqETeiCMnDc4QrFr7bPfAAHLV7W23c3YwaHRRMW7TH1fb20iEtwLvddAZ93wiLnW3q12vh+cvDE7wFwc3m5yFs5EHuUJqohuql5LJSZaPoHisIcHkap0zsnKf2kXfpRm2idQn+AcdGTci1TqRozO6PXI+xdE76IueI7SCN4xcHRW+pgXtlIYzg59MzlEQue2mHPYyuU0fHIJdxLRJzo0Ko2tQFBYqzNVuj5Gjp0P+AH4A7VUiJ+2iHNhBN2R3asnkzqxI/xxWEHIWglFHZ7dw89tgKKwmKq/NIXI6+vRk3uQS5gXNaG66TMgFPquP7qiLfC27C+SIWAlKq0rIyWH/3J7FnEq9U6vV8S+jklj1yXHkg6JmGeUuA8gpzKksn9QQuYw+6KRz0yFNtiOkMrKXCdVzyOGAFl4poDvq4LOhWquqBM+vRMoofjWExRDE+Qg8YeQayT4i52EDNdGohQ78B/QMXFQVyrhos6eNGhH0O1mnPy7pjpCMHDULuLNygA5jxyGUA7nE6AtXGSE5nPfIcUKujVqOpiVbHhPHkIutYSZ+B0fh6oAFkcPdvLhakmtjkcNVjN8OyXlWWCOdVMl3xye37BF/Mjl3MnIDVLFFyTWxA6KsPLlChFx30jzHk2vcjjxXmKK0diOlVQ7JeZPWcwsjp0xAThHJoTtCnj60EGltK09ObCHaNJhg9RxuW8chB0HctO/xsxUTYGtoW+JWTJBh7Qg2+d+Cw7CPkWVI3GIQclmUU3Kwj5G1iyNX06FeCV4Qnar7mByRG5Qc3QWJkMPH8bsChBw2OtgrofPvpYjpgRh9VLCw39hW2doR8OMFKXbtCAleLZAFuUiCyYO1IxbqCSN3tdZF/qrWqFX/JzFPOIwhsnjC6FnUAk9YtXC1B6d2ZeYJo6wbesLzWq9EjtQZ0pyjLxPH8Cj6InHr9iBL9BXMjiDksNjHFZtDUeDoq0a69vjoy9heQPQ1f3LVgFwP977hiN9ErLDLT++tdFZLivgj5NpADvePbJcAhYpqvU7QV7Jy5DpNevYCipRquJdJNXFFV2vgfXVRDeZ1Q3K4IowlV0BH6jSHuoghvgDcy4SamrPVJYe7Ng8ku9FB4FyVomiiTFc98Xy5WUfZpsrqOdVDn3uRei6GnIpKKGoV0GdnHRc3LfbKkqvigYEuqsoRONL5iBoUE/knuJcd571qSE726/jTNuly4+bP6Y9qpJcJp12EjoxN1HB7ogC5WpRcbf7kJhvBybrOJm0hmnt7lF6ddnJguV7vuuRDt33mB+Rwn4iLMhIZh2i47p7Fuim0k24XkaNn73fQCcn4xhnyCik5hJIOD5H+uW4Xta2zHcERZ4HBTkJ2GVY0DTYmIibbxyhcrJVtTATro46QG5jcmaFrzXq93nM0m+5cpCFpsbjreV5DN2xDRuR6ZXp229Z7zbpfRFZRsw0b5FhfZGcvlx3Da3faXs/AJyQXg46za9eIuMh2SRp1a7BYK9svKqOcOcZzG6mmnnBRJc4k95Ymygzs8erEiwu9RDqwLTiV0YyP8iaSsCrDhMMRp5LL+HBtN+wtTS76Spod0SdeXHJleyvnlWQl5+bkJiHnoWbTl3NyY5MzcWelmpMbm5wGHUU3lhz1Vma6Fpg49hU7w7p/gmeBkNn7o14t4J3K614LbL4bFJnYn9vVEo7aJd/rkIk6xnwvYx5pzm+SyCiYivTP8W+S6CQCw1Mg2Cb02R535jdJIOeDOev9vmS+zpj5XKZ2NzI7gh/q1PBLNp1O0w+mdiZUtsv59tJ8yamy4ns9OZ6cZDW9no+jgZxc7BtzVvIbcwXLispzcjN7pzont5LkhMGs61uhVAo7bMRuinK6XFyhdEH7VEPnlNDxFtkRnNpsR3AwtRFyvq+rnNjXNZ7cnESeeGuivJwuF0kwgMu00kvGGGLKDaCSVnpZlvlzjBwvT1ldiJJLr2KWLG7NyeXkFkPu/ySKcxRFDbS6AAAAAElFTkSuQmCC'
        f = urllib.request.urlopen(imgURL)
        # read the image file in a numpy array
        face = mpimg.imread(f)
        fct.drawPicture(face)

    print("Next (1, 2, 3, 4, 5, exit) : ")
    choice = input()
print("BYE, BYE")

# ----------- BONUS -------------------
# nbDim = face.shape
# nbLine = nbDim[0]
# nbCol = nbDim[1]
#origin : This parameter is used to place the [0, 0] index of the array in the upper left or lower left corner of the axes.
# plt.imshow(face, cmap=plt.cm.gray, origin="lower")
# plt.show()
# plt.imshow(face, cmap=plt.cm.gray, origin="upper")
# plt.show()




print("END")
