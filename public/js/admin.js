window.addEventListener("load", function () {
  document.querySelector(".nav--loading").style.display = "none";
});

const url = "http://localhost:8080";
var header = `<div class="header--logo">
</div>
<h1 style="font-size: 30px;">Nhóm 04</h1>
<div class="header--tool">
</div>`;
var sidebar = `<div class="content--sidebar--items">
<a href="adminpage.html" class="content--sidebar--item adminpage"><i class="fa-regular fa-gauge-max"></i>Data</a>
<a href="adminpage2.html" class="content--sidebar--item productpage"><i class="fa-regular fa-book"></i>Content</a>
</div>`;
$(function () {
  $(".header").html(header);
});
function openPopup(ele) {
  document.getElementById(ele).style.display = "block";
}
function closePopup(ele) {
  document.getElementById(ele).style.display = "none";
}
document.getElementsByClassName("popup--board").item(0).addEventListener("click", function (event) {
  event.stopPropagation();
});

