// Handle navbar dropdown
const navDropdown = $('.nav-item.dropdown');
navDropdown.on('mouseover', function() {
    $(this).addClass('show');
    $(this).find('.dropdown-menu').addClass('show');
});
navDropdown.on('mouseout', function() {
    $(this).removeClass('show');
    $(this).find('.dropdown-menu').removeClass('show');
});
console.log(navDropdown);
