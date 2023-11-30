window.addEventListener("load", function (event) {
  document
    .getElementsByClassName("adminpage")
    .item(0)
    .classList.add("content--sidebar--item__active");

  // Show tong don hang
  var requestOptions = {
    method: "GET",
    redirect: "follow",
  };

  fetch("http://localhost:8080/api/showSoLuongDonhang", requestOptions)
    .then((response) => response.text())
    .then((result) => {
      document.getElementById("tongDH").innerHTML = result;
    })
    .catch((error) => console.log("error", error));

  // show tong khach hang
  var requestOptions = {
    method: "GET",
    redirect: "follow",
  };

  fetch("http://localhost:8080/api/showSoLuongKH", requestOptions)
    .then((response) => response.text())
    .then((result) => {
      document.getElementById("tongKH").innerHTML = result;
    })
    .catch((error) => console.log("error", error));

  // show tong doanh thu
  var requestOptions = {
    method: "GET",
    redirect: "follow",
  };

  fetch("http://localhost:8080/api/showDoanhThu", requestOptions)
    .then((response) => response.text())
    .then((result) => {
      console.log(result);
      document.getElementById("tongDoanhThu").innerHTML = formatMoneyVND(parseFloat(result));
    })
    .catch((error) => console.log("error", error));

  // show don hang moi
  var requestOptions = {
    method: "GET",
    redirect: "follow",
  };

  fetch("http://localhost:8080/api/showDonHangMoi", requestOptions)
    .then((response) => response.json())
    .then((result) => {
      var html = ``;
      for (let i = 0; i < result.length; i++) {
        html +=
          `<div class="content--order--item">
              <div class="content--order--item--column">` +
          result[i].username +
          `</div>
              <div class="content--order--item--column">
                ` +
          result[i].orderDate +
          `
              </div>
              <div class="content--order--item--column">` +
          result[i].soluong +
          `</div>
              <div class="content--order--item--column">` +
              formatMoneyVND(result[i].orderTotal) +
          `</div>
              <div class="content--order--item--column">` +
          result[i].order_status +
          `</div>
            </div>`;
      }
      document.getElementById("content--order--items").innerHTML = html;
    })
    .catch((error) => console.log("error", error));
});
