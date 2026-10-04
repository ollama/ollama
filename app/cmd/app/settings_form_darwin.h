#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

// OLSettingsForm lays out settings the way macOS settings windows do: labels
// ending in colons are right-aligned in one column, and controls with their
// descriptions are left-aligned in the next, in groups separated by space.
@interface OLSettingsForm : NSObject
@property(nonatomic, readonly) NSGridView *gridView;

// width is the form's total width, and labelWidth the width of its label
// column. Descriptions wrap to fit the rest.
- (instancetype)initWithWidth:(CGFloat)width labelWidth:(CGFloat)labelWidth;

// addRowWithLabel:view: adds a row. A nil label continues the group above.
- (NSGridRow *)addRowWithLabel:(nullable NSString *)label view:(NSView *)view;

// addDescription: adds explanatory text under the previous row's control,
// aligned with a checkbox's title.
- (NSTextField *)addDescription:(NSString *)text;

// addDescriptionView: adds view where addDescription: would add text.
- (NSGridRow *)addDescriptionView:(NSView *)view;

// controlWidth is the width of the control column.
@property(nonatomic, readonly) CGFloat controlWidth;

// addGroupSpacing separates the next row from the group above.
- (void)addGroupSpacing;

// addSeparator adds a line across the form between groups.
- (void)addSeparator;
@end

// OLCheckbox returns a checkbox that sends action to target when clicked.
NSButton *OLCheckbox(NSString *title, id _Nullable target, SEL _Nullable action);

// OLPushButton returns a standard push button.
NSButton *OLPushButton(NSString *title, id _Nullable target, SEL _Nullable action);

// OLDescriptionLabel returns small, secondary text that wraps.
NSTextField *OLDescriptionLabel(NSString *text);

// OLControlWithAccessory places accessory at the trailing edge of the
// control's row, like the Manage… buttons in settings windows.
NSView *OLControlWithAccessory(NSView *control, NSView *accessory);

// OLHorizontalStack returns views side by side, aligned on their baselines.
NSStackView *OLHorizontalStack(NSArray<NSView *> *views);

NS_ASSUME_NONNULL_END
