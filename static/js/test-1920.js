
getNowFormatDate();

//獲取當前時間
function getNowFormatDate() {
	var date = new Date();
	var year = date.getFullYear();
	var month = date.getMonth() + 1;
	var strDate = date.getDate();
	var Hour = date.getHours();
	var Minute = date.getMinutes();
	var Second = date.getSeconds();
	var show_day = new Array('星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六');
	var day = date.getDay();
	if (Hour < 10) {
		Hour = "0" + Hour;
	}
	if (Minute < 10) {
		Minute = "0" + Minute;
	}
	if (Second < 10) {
		Second = "0" + Second;
	}
	if (month >= 1 && month <= 9) {
		month = "0" + month;
	}
	if (strDate >= 0 && strDate <= 9) {
		strDate = "0" + strDate;
	}
	var currentdate = '<div><p>' + year + '年' + month + '月' + strDate + '號</p><p>' + show_day[day] + '</p></div>';
	var HMS = Hour + ':' + Minute + ':' + Second;
	var temp_time = year + '-' + month + '-' + strDate + ' ' + HMS;
	$('.nowTime li:nth-child(1)').html(HMS);
	$('.nowTime li:nth-child(2)').html(currentdate);
	setTimeout(getNowFormatDate, 1000);
}
var resourceDataType = $('.data-label li.active').data('type');
function urlType() {
	resourceDataType = $('.data-label li.active').data('type');
	if (resourceDataType == 1) {
		init_myChart3(legal_person_data);
		$('.com-screen-content .use-data').html(Tpl1);
	} else if (resourceDataType == 2) {
		init_myChart3(people_data);
		$('.com-screen-content .use-data').html(Tpl2);
	} else if (resourceDataType == 3) {
		init_myChart3(picture_data);
		$('.com-screen-content .use-data').html(Tpl3);
	}
}
