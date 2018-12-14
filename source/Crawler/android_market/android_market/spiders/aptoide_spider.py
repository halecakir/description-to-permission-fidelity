import scrapy

class AptoideSpider(scrapy.Spider):
    name = "aptoide-spider"

    def start_requests(self):
        urls = ["https://en.aptoide.com/apps/latest/more"]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parseLatestApps)

    def parseLatestApps(self, response):
        #Application links
        app_pages = response.xpath('//*[@class="bundle-item__info__span bundle-item__info__span--big"]/a/@href').extract() 
        for link in app_pages:
            yield scrapy.http.Request(url=link, callback=self.parseApp)

    def parseApp(self, response):
        app_name = response.xpath('//*[@class="header__title"]/text()').extract_first()
        app_description = response.css('div.view-app__description *::text').extract()
        app_permissions = response.xpath('//*[@class="app-permissions__row"]/span/text()').extract()

