# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as BS
import requests
import os
import argparse

cookie_file = None # cookie txt file. If are not paid member, it's unnecessary.
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}


def parseArgs():
    parser = argparse.ArgumentParser(description='Choose download task for image collection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', required=True, type=str,
                        help='download task name')
    args = parser.parse_args()
    return args


def loadCoockies(cookie_file):
    '''
    load cookies from txt file.
    :param cookie_file: txt file containing cookies information
    :return cookies: cookies information in dict format
    '''
    cookies = {}
    if cookie_file == None:
        return cookies
    
    with open(cookie_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(':')
            cookies[line[0]] = line[1]
    return cookies
cookies = loadCoockies(cookie_file)


# 解析下载Actress的信息
def downloadActressDetail(idol_index):
    '''
    Using actress index to scrap information from jav database.
    The actress information and avatar are saved in related directory.
    根据演员索引进行演员资料的下载，相应的信息和头像存储在对应的文件夹中。
    :param idol_index: int number
    :return None:
    '''
    root_dir = '../data/actress_details/'
    url = 'https://xxx.xcity.jp/idol/detail/?id=%d&style=simple' % idol_index
    web_data = requests.get(url)
    soup = BS(web_data.text, 'lxml')
    
    # 获取姓名，建立文件夹
    act_info = []
    # try to get act name, return when error happnens
    try:
        act_name = soup.select('#avidolDetails > div > h1')[0]
        act_name = act_name.text
    except Exception as e:
        print('Can not get act name in idol %d , erros happens.' % idol_index)
        print('And the error is ', e)
        return
    
    # make directory or return
    if '/' in act_name:
        act_name = act_name.replace('/', ',')
    if not os.path.exists(os.path.join(root_dir, act_name)):
        os.makedirs(os.path.join(root_dir, act_name))
    else:
        return
    
    # in some pages, there is favorite option; but in some others, there isn't.
    temp = soup.select('#avidolDetails > div > div.frame > dl > dd')
    if len(temp) == 7:
        stars = '0'
    else:
        stars = temp[0].findAll('span')[1].text
        temp = temp[1:]
    temp = [tag.text for tag in temp]
    # parse actress information
    # actrually, to avoid some unkown page elements change, using contents to find information
    # rather than index is a better way.
    birthday = temp[0].replace('Date of birth', '')
    birthday = birthday.replace('/', '-') # 将日期设置为更通用的格式
    blood = temp[1].replace('Blood Type', '')
    city = temp[2].replace('City of Born', '')
    height = temp[3].replace('Height', '')
    size = temp[4].replace('Size', '')
    hobby = temp[5].replace('Hobby', '')
    skill = temp[6].replace('Special Skill', '')
    
    # save information
    act_info.append('Actress Name:' + act_name)
    act_info.append('Birthday:' + birthday)
    act_info.append('Blood:' + blood)
    act_info.append('City:' + city)
    act_info.append('Height:' + height)
    act_info.append('Size:' + size)
    act_info.append('Hobby:' + hobby)
    act_info.append('Skill:' + skill)
    act_info.append('Stars:' + stars)
    act_info = [line.replace('\n', '') for line in act_info]
    act_info = [line+'\n' for line in act_info]
    txt_file = '%s.txt' % act_name
    with open(os.path.join(root_dir, act_name, txt_file), 'w') as f:
        f.writelines(act_info)
    
    # save avatar
    try:
        avatar_url = 'https://fs.xcity.jp/imgsrc/image/person/%d.jpg' % idol_index
        avatar_file = '%s.jpg' % act_name
        avatar_img = requests.get(avatar_url)
        with open(os.path.join(root_dir, act_name, avatar_file), 'wb') as f:
            f.write(avatar_img.content)
    except Exception:
        print('Can not download the avatar of %d' % idol_index)


def downloadActressDetails():
    '''
    To download all actress details.
    下载所有的女优信息。
    :param None:
    :return None:
    '''
    for idol_index in range(1, 10000): # you find update the idol index range by view the website
        try:
            downloadActressDetail(idol_index)
        except Exception as e:
            print('Unexpected error when downloading actress details of actress %d' % idol_index)
            print('And the error is ', e)
            continue
        print('Actress detail of actress %d downloaded.' % idol_index)


# 解析下载电影信息
def downloadMovieDetail(movie_index):
    '''
    Using movie index to download movie information and post.
    The information and post are saved in related directory.
    使用影片序号下载详细资料和海报，相应信息存储在对应的文件夹中。
    :param movie_index: movie index, int number
    :return None:
    '''
    root_dir = '../data/movie_details/'
    url = 'https://xxx.xcity.jp/avod/detail/?id=%d' % movie_index
    web_data = requests.get(url, cookies=cookies, headers=headers)
    soup = BS(web_data.text, 'lxml')
    
    # parse movie information
    # part 0: movie name
    title = soup.select('#program_detail_title')[0]
    title = title.text
    if len(title) == 0:
        print('Can not get movie name in movie index %s' % movie_index)
        return
    
    # part1
    temp = soup.select('#avodDetails > div > div.frame > div.content > div > ul.profileCL > li')
    if len(temp) == 5:
        stars = '0'
    else:
        stars = temp[0].findAll('span')[1].text
        temp = temp[1:]
    # 销售日期,卡司，商标,发行商,所属系列,风格
    # sale date, actress, label, maker, serie, genres
    sale_date = temp[0].text.replace('Sales Date', '')
    sale_date = sale_date.replace('/', '-')
    # multi actresses
    cast = temp[1].findAll('a')
    cast = [c.text for c in cast]
    cast = ','.join(cast)
    label_maker = temp[2]
    label = label_maker.find('span',{'id':'program_detail_label_name'}).text
    maker = label_maker.find('span',{'id':'program_detail_maker_name'}).text
    serie = temp[3].text.replace('Series', '')
    # multi genres
    genres = temp[4].findAll('a',{'class':'genre'})
    genres = [genre.text for genre in genres]
    genres = ','.join(genres)
    
    # part2
    temp = soup.select('#avodDetails > div > div.frame > div.content > div > ul.profile > li')
    temp = [tag.text for tag in temp]
    # 导演,番号,时长,释出日期,剧情描述
    # director, movie number, length, release date, description
    director = temp[0].replace('Director' ,'')
    item_id = temp[1].replace('Item Number' ,'')
    length = temp[2].replace('Running Time' ,'')
    release_date = temp[3].replace('Release Date' ,'')
    release_date = release_date.replace('/', '-')
    description = temp[4].replace('Description' ,'')
    
    # save information
    movie_info = []
    movie_info.append('Index:' + str(movie_index))
    movie_info.append('ID:' + item_id)    
    movie_info.append('Title:' + title)
    movie_info.append('Date:' + sale_date)
    movie_info.append('Release Date:' + release_date)
    movie_info.append('Length:' + length)
    movie_info.append('Director:' + director)
    movie_info.append('Maker:' + maker)
    movie_info.append('Label:' + label)
    movie_info.append('Series:' + serie)
    movie_info.append('Genres:' + genres)
    movie_info.append('Stars:' + stars)
    movie_info.append('Actress:' + cast)
    movie_info.append('Description:' + description)
    movie_info = [line.replace('\n', '') for line in movie_info] # remove return
    movie_info = [line.replace('\t', '') for line in movie_info] # remove tab
    movie_info = [line+'\n' for line in movie_info]
    
    # make directory or return
    if not os.path.exists(os.path.join(root_dir, item_id)):
        os.makedirs(os.path.join(root_dir, item_id))
    else:
        return
    movie_file = os.path.join(root_dir, item_id, ('%d.txt' % item_id))
    with open(movie_file, 'w') as f:
        f.writelines(movie_info)
    
    # download movie post
    try:
        post_url = soup.select('#avodDetails > div > div.frame > div.photo > p > a')[0]
        post_url = post_url['href']
        if post_url.startswith('//'):
            post_url = 'https:' + post_url
        else:
            post_url = 'https://' + post_url
        # 下载保存
        post_file = post_url.split('/')[-1]
        post_file = os.path.join(root_dir, item_id, post_file)
        post_img = requests.get(post_url, cookies=cookies, headers=headers)
        with open(post_file, 'wb') as f:
            f.write(post_img.content)
    except Exception:
        print('Can not download post image of movie %d' % movie_index)


def downloadMovieDetails():
    '''
    Download all movie details.
    下载所有影片的信息。
    :param None:
    :return None:
    '''
    for movie_index in range(1, 160000): # you can change movie index range
        try:
            downloadMovieDetail(movie_index)
        except Exception as e:
            print('Unexpected error when downloading movie details of movie %d' % movie_index)
            print('And the error is ', e)
            continue
        print('The information of %s have been downloaded.' % movie_index)

# 解析下载电影截图
def pruneURL(string):
    '''
    Add https:// and remove unuseful element in image url link.
    为图片url链接天机https://，并去除不必要的信息。
    :param string: url link to process
    :return string: url link processed.
    '''
    if string.startswith('//'):
        string = 'https:' + string
    else:
        string = 'https://' + string
    index = string.index('?width')
    string = string[:index]
    return string


def parseMovieSample(soup):
    '''
    parse sample image links in soup object.
    解析soup中的sample图片链接。
    :param soup: soup object of related movie
    :return urls: image url links, a string list
    '''
    urls = soup.select('#itemCol > div.sub_container > div.ondemand_imagearea')
    if len(urls) == 0:
        return None
    urls = urls[0].findAll('a')[1:]
    urls = [url['href'] for url in urls]
    urls = [pruneURL(url) for url in urls]
    return urls


def parseMovieAction(soup):
    '''
    parse action image links in soup object.
    解析soup中的action图片链接。
    :param soup: soup object of related movie
    :return urls: image url links, a string list
    '''
    urls = soup.select('#photo > div.scenes')
    if len(urls) == 0:
        return None
    urls = urls[0].findAll('img')
    urls = [url['src'] for url in urls]
    urls = [pruneURL(url) for url in urls]
    return urls


def parseMovieGravure(soup):
    '''
    parse gravure image links in soup object.
    解析soup中的gravure图片链接。
    :param soup: soup object of related movie
    :return urls: image url links, a string list
    '''
    urls = soup.select('#gravure > div.scenes')
    if len(urls) == 0:
        return None
    urls = urls[0].findAll('img')
    urls = [url['src'] for url in urls]
    urls = [pruneURL(url) for url in urls]
    return urls


def saveImageLinks(save_path, links):
    '''
    Save image links in txt file.
    将解析得到的图片链接保存在文本文件中。
    :param save_path: txt file path to save image links
    :param links: image links
    :return None:
    '''
    links = [line+'\n' for line in links]
    with open(save_path, 'w') as f:
        f.writelines(links)


'''
IMPORTANT INFORMATION!!!
In this website, you are free to get sample images, but action image and gravure image
are not free. To download them, you need to be paid member.
此网站中的sample图片可以免费获取，但是action和gravure部分的图片需要付费才能浏览。
'''
def downloadImageLinks():
    '''
    Download image links using movie index, and image links are saved in related directory.
    根据影片序号下载影片的索引，并将图片链接存储在对应的文件夹中。
    :param None:
    :return None:
    '''
    root_dir = '../data/movie_image_links/'
    for movie_id in range(1, 160000):
        # make directory or return
        if not os.path.exists(os.path.join(root_dir, str(movie_id))):
            os.makedirs(os.path.join(root_dir, str(movie_id)))
        else:
            return
        
        url = 'https://xxx.xcity.jp/avod/detail/?id=%d' % movie_id
        try:
            web_data = requests.get(url, headers=headers)
            soup = BS(web_data.text, 'lxml')
        
            sample = parseMovieSample(soup)
            action = parseMovieAction(soup)
            gravure= parseMovieGravure(soup)
            if sample != None:
                saveImageLinks(os.path.join(root_dir, str(movie_id), 'sample.txt'), sample)
            if action != None:
                saveImageLinks(os.path.join(root_dir, str(movie_id), 'action.txt'), action)
            if gravure != None:
                saveImageLinks(os.path.join(root_dir, str(movie_id), 'gravue.txt'), gravure)
            print('Movie %d image links downloaded.' % movie_id)
        except Exception as e:
            print('Movie %d image links downloading meets error.' % movie_id)
            print('And the error is ', e)
            

def downloadImages(task='sample'):
    '''
    Download movie images from downloaded movie image links.
    根据已下载的影片图片链接下载影片图片。
    :param task: whicn type of image you want to download, sample, action or gravure
    :return None:
    '''
    input_dir = '../data/movie_image_links/'
    output_dir = '../data/movie_images_of_%s' % task
    movie_list = os.listdir(input_dir)
    movie_list.sort()
    for movie in movie_list:
        # no related image links txt file
        if not os.path.exists(os.path.join(input_dir, movie, '%s.txt' % task)):
            continue
        # jump downloaded movies
        if not os.path.exists(os.path.join(output_dir, movie)):
            os.makedirs(os.path.join(output_dir, movie))
        else:
            continue
        # load image links
        with open(os.path.join(input_dir, movie, '%s.txt' % task)) as f:
            links = f.readlines()
        links = [line.strip() for line in links]
        # download images
        for link in links:
            try:
                image = link.split('/')[-1]
                image = os.path.join(output_dir, movie, image)
                img = requests.get(link, cookies=cookies, headers=headers)
                with open(image, 'wb') as f:
                    f.write(img.content)
            except Exception as e:
                print('Something wrong with image link %s in moive %s' % (link, movie))
                print('And the error is ', e)
        print('Movie %s downloaded.' % movie)


if __name__ == '__main__':
    args = parseArgs()
    if args.task == 'download_actress_details':
        downloadMovieDetails()
    elif args.task == 'download_movie_details':
        downloadMovieDetails()
    elif args.task == 'download_image_links':
        downloadImageLinks()
    elif args.task == 'download_images':
        downloadImages()
    else:
        print('You should choose the right task.')
    print('Download task %s finished.' % args.task)
    
