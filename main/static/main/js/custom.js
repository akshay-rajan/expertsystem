// Handle navbar dropdown
const navDropdown = $('.nav-item.dropdown');

navDropdown.on('mouseover', function() {
    $(this).addClass('show');
    $(this).find('.dropdown-menu').addClass('show');
});

// Add mouseout event to only close dropdown if mouse leaves the dropdown and the menu
navDropdown.on('mouseout', function(e) {
    // Check if mouse left the entire dropdown (including the menu)
    if (!$(e.relatedTarget).closest('.navbar').length) {
        $(this).removeClass('show');
        $(this).find('.dropdown-menu').removeClass('show');
    }
});
$('.navbar').on('mouseout', function(e) {
    if (!$(e.relatedTarget).closest('.navbar').length) {
        navDropdown.removeClass('show');
        navDropdown.find('.dropdown-menu').removeClass('show');
    }
});
