function init(){
    $('.awesome-tooltip').tooltip({
        placement: 'left'
    });   
    $(window).bind('scroll',function(e){
        dotnavigation();
    });
    function dotnavigation(){  
        var numSections = $('section').length;
        
        $('#dot-nav li a').removeClass('active').parent('li').removeClass('active');     
        $('section').each(function(i,item){
        var ele = $(item), nextTop;        
        if (typeof ele.next().offset() != "undefined") {
        nextTop = ele.next().offset().top;
        }
        else {
        nextTop = $(document).height();
        }
        if (ele.offset() !== null) {
        thisTop = ele.offset().top - ((nextTop - ele.offset().top) / numSections);
        }
        else {
        thisTop = 0;
        }
        var docTop = $(document).scrollTop();
        if(docTop >= thisTop && (docTop < nextTop)){
        $('#dot-nav li').eq(i).addClass('active');
        }
        // if ((docTop >= nextTop) && (i == 0) && ($('#trendChart').highcharts() == null)) {
        //     plotTrend()
        // }
        // if ((docTop >= nextTop) && (i == 1) && ($('#areaChart').highcharts() == null)) {
        //     plotBar()
        // }
        // if ((docTop >= nextTop) && (i == 2) && ($('#mapChart').highcharts() == null)) {
        //     plotArea()
        // }
        // if ((docTop >= nextTop) && (i == 3) && ($('#barChart').highcharts() == null)) {
        //     plotMap()
        // }
        });
    }
}

$(document).ready(init());
$(Highcharts.charts).each(function(i,chart){
    var height = chart.renderTo.clientHeight; 
    var width = chart.renderTo.clientWidth; 
    chart.setSize(width, height); 
  });