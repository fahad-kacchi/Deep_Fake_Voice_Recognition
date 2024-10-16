// document.addEventListener("DOMContentLoaded", function () {
//     // Function to handle the Get started button click
//     document.querySelector("button").addEventListener("click", function () {
//         alert("Get started button clicked!");
//         // Optionally, you can add smooth scrolling to a certain section
//         // window.scrollTo({
//         //     top: document.querySelector('.target-section').offsetTop,
//         //     behavior: 'smooth'
//         // });
//     });
// });

function displayFileName() {
    const input = document.getElementById('audio-upload');
    const fileName = document.getElementById('file-name');

    if (input.files.length > 0) {
        fileName.textContent = input.files[0].name;
    } else {
        fileName.textContent = 'No file chosen';
    }
}
