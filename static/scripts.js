document.getElementById("upload-form").addEventListener("submit", function () {
    document.getElementById("loader").style.display = "block";
});

function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
        const preview = document.querySelector(".thumb img");
        if (preview) {
            preview.src = reader.result;
        }
    }
    reader.readAsDataURL(event.target.files[0]);
}