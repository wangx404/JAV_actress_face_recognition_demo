Script downloader.py in this folder is used to download images from JAV database.

## Usage Description

You can run this script with command `python downloader.py --task xxx`, in which `xxx` is the download task. Tasks are listed below.

- `download_actress_details`: download details of all actresses. The details include name, birthday, height and other information of the actress. You can view [this example](https://xxx.xcity.jp/idol/detail/?id=1&style=simple) to see what information are includes.

- `download_movie_details`: download details of all movies. THe details include item id, title, actress, release date and other information of the movie. You can view [this example](https://xxx.xcity.jp/avod/detail/?id=1) to see what information are included.

- `download_image_links`: download image links of all movies. In the movie page, you can see that sample images, action images and gravure imames. This option is to download these image links in movie page. But you need to notice that, the action images and gravure images are not free to get. So if you want to download them in this and next step, you need to be a paid member.

- `download_images`: download images from image links downloaded. You can choose whick type of image to download in the script. After setting the value (sample, action, gravure) of the variable of function `downloadImages`, you can download opposite images.

## Notice

1.  Hereby, I recommend you to run this script in order, though only step three and four are related.

2.  In step three and four, files are saved in directory named after movie index. So when the script is done, you can change the directory name from movie index to item id. In this way, you can resort the images clearly.

3. To speed up downloading speed, you can write a multi-thread function to invoke other functions. But I really don't recommend you to do this. Because this would increase the server pressure. Please be patient when you don't pay for this.

4.  In some movies, there are multi actresses. So after downloading all there details and images, you may need to write a script to delete or put aside these files.

5.  You can write your own script to download images from other websites. The process is pretty similar.

**P.S.Even though these adult videos are legal in Japan and other countries, It's illegal to spread these videos and pictures in China. So please be careful, and not save them in servers in China.**
