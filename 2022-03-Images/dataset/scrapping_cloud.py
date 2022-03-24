from scrapping_util import get_page
import requests
import sys
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PERSO\\")
from ara_commons.ara_file import get_dir_files
from os import getcwd


def get_img_urls_liste(page, verbose=0):
    """Search all article URLs in the page

    Args:
        page (BeautifulSoup): _description_
        verbose (int, optional): log level. Defaults to 0.

    Raises:
        AttributeError: if page is missing

    Returns:
        list(str): list of IMG URL
    """
    if page is None:
        raise AttributeError("page expected")

    # récupérer la liste des noms de pokémon
    liste_urls =set()

    try:
        for lien in page.findAll('a', {'title': 'Download photo'}):
            link = lien.get('href')
            liste_urls.add(link)
    except Exception as error:
        pass

    try:
        for lien in page.findAll('img', {'class': 'YVj9w'}):
            link = lien.get('src')
            liste_urls.add(link)
    except Exception as error:
        pass

    try:
        for lien in page.findAll('img', {'class': 'MosaicAsset-module__thumb___klD9E'}):
            link = lien.get('src')
            liste_urls.add(link)
    except Exception as error:
        pass
        
    if verbose>1:
        print(len(liste_urls), "URLs found")
    return liste_urls


def save_img(url, prefix="", path="", verbose=0):
    if url is None or len(url)==0:
        raise AttributeError("URL expected")

    response = requests. get(url)
    
    contentType = response.headers['content-type']
    if "image" in contentType:
        f_temp = get_dir_files(path)
        idx = 0
        if f_temp is not None:
            idx = len(f_temp)

        ext = contentType.split("/")[-1]
        idx_str = str(idx)
        file_name = prefix+idx_str+"."+ext

        file = open(path+file_name, "wb")
        file. write(response.content)
        file. close()


def scrape(save_path, verbose = 0):
    """Return the url liste to scrapt

    Args:
        verbose (int, optional): log level. Defaults to 0.

    Returns:
        tuple(Journal, list[Article]): Papers and article list
    """
    to_request_urls = { 
                        # "cloudy"  : ['https://unsplash.com/s/photos/clouds',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1316024691&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fclouds&utm_medium=affiliate&utm_source=unsplash&utm_term=clouds%3A%3A%3A',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=false&assetid=1316024691&page=2',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=false&assetid=1316024691&page=7',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1340824257&page=2',                                     
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1340824257&page=5',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=184103864&page=2',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=184103864&page=8',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1325888847&page=2',
                        #              'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1325888847&page=8'],
                        # "shine" : [
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1131146226&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fsun&utm_medium=affiliate&utm_source=unsplash&utm_term=sun%3A%3A%3A',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=637284498&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fshine-beach&utm_medium=affiliate&utm_source=unsplash&utm_term=shine+beach%3A%3A%3A',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=637284498&page=2',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=637284498&page=5',
                        #            'https://www.istockphoto.com/fr/search/more-like-this/1222094964?assettype=image&excludenudity=true&alloweduse=availableforalluses&mediatype=photography&phrase=shine%20beach',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1131146226&page=2',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1131146226&page=4',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1131146226&page=8',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1087673356&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fsun-shine&utm_medium=affiliate&utm_source=unsplash&utm_term=sun+shine%3A%3A%3A',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1087673356&page=2',
                        #            'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1087673356&page=5'
                                   
                                   
                                #    ],
                        "sunrise" : ['https://unsplash.com/s/photos/sunrise',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=825148240&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fsunrise&utm_medium=affiliate&utm_source=unsplash&utm_term=sunrise%3A%3A%3A',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=825148240&page=2',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=825148240&page=5',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1068270866&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fsunrise&utm_medium=affiliate&utm_source=unsplash&utm_term=sunrise%3A%3A%3A',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1068270866&page=2',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1068270866&page=5',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1124629093&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fsunrise&utm_medium=affiliate&utm_source=unsplash&utm_term=sunrise%3A%3A%3A',
                                     'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=523384127&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fsunrise&utm_medium=affiliate&utm_source=unsplash&utm_term=sunrise%3A%3A%3A'
                                    # ],
                        # "rain" : ['https://unsplash.com/s/photos/raining',
                        #           'https://unsplash.com/s/photos/rain',
                        #           'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=503284599&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fraining&utm_medium=affiliate&utm_source=unsplash&utm_term=raining%3A%3A%3A',
                        #           'https://www.istockphoto.com/fr/search/2/image?excludenudity=true&alloweduse=availableforalluses&phrase=raining&sort=best',
                        #           'https://www.istockphoto.com/fr/search/2/image?excludenudity=true&alloweduse=availableforalluses&phrase=raining&sort=best&page=2',
                        #           'https://www.istockphoto.com/fr/search/2/image?excludenudity=true&alloweduse=availableforalluses&phrase=raining&sort=best&page=3',
                        #           'https://www.istockphoto.com/fr/search/2/image?excludenudity=true&alloweduse=availableforalluses&phrase=raining&sort=best&page=4',
                        #           'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1257951336&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fraining&utm_medium=affiliate&utm_source=unsplash&utm_term=raining%3A%3A%3A',
                        #           'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1257951336&page=2',
                        #           'https://www.istockphoto.com/fr/search/search-by-asset?searchbyasset=true&assettype=image&excludenudity=true&alloweduse=availableforalluses&assetid=1257951336&page=6',
                        #           'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1148404365&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fraining&utm_medium=affiliate&utm_source=unsplash&utm_term=raining%3A%3A%3A',
                        #           'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=1143045785&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fraining&utm_medium=affiliate&utm_source=unsplash&utm_term=raining%3A%3A%3A',
                        #           'https://www.istockphoto.com/fr/search/search-by-asset?affiliateredirect=true&assetid=155278867&assettype=image&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fraining&utm_medium=affiliate&utm_source=unsplash&utm_term=raining%3A%3A%3A'
                                  ]
                        }
    # Parcours de la liste des URLs
    for type_, urls_list in to_request_urls.items():
        for end_point in urls_list:
            try:
                page = get_page(end_point)
                liste_urls = get_img_urls_liste(page, verbose=verbose)

                if liste_urls is not None and len(liste_urls) > 0:
                    for url_img in liste_urls:
                        try:
                            save_img(url_img, prefix=type_, path=save_path, verbose=0)
                        except Exception as error:
                            pass
            except Exception as error:
                print(error)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
   
    url = 'https://media.istockphoto.com/photos/woman-hand-holding-cotton-wool-on-cloud-sky-background-the-of-the-picture-id1054969220?k=20&m=1054969220&s=612x612&w=0&h=V-vZCbb5Bulxca9QtCnnl90svfzm8pkR9YKl7HPWTEc='
    response = requests. get(url)
    contentType = response.headers['content-type']
    print(contentType)

    # cloud_path = "https://unsplash.com/images/nature/cloud"
    # Récupère le répertoire du programme
    file_path = getcwd()
    save_path = file_path +  "\\simplon\\2022-03-Images\\dataset_800\\"
    get_dir_files(save_path)

    # scrape(save_path, verbose=1)
    scrape(save_path, verbose=1)
    print("END")

            

