# How to scrape images

- Install python library
```
$ pip install google_images_download
```
- Install google-chrome
- Download chromedriver associated with the installed version of google-chrome
- Setup the categories in `download.json`
- Now you can download the images for each category
```
$ googleimagesdownload -cf download.json
```

