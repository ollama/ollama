#ifndef MENU_H
#define MENU_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    char *label;
    int enabled;
    int separator;
} menuItem;

// TODO (jmorganca): these need to be forward declared in the webview.h file
// for now but ideally they should be in this header file on windows too
#ifndef WIN32
int menu_get_item_count();
void *menu_get_items();
void menu_handle_selection(char *item);
#endif

#ifdef __cplusplus
}
#endif

#endif
