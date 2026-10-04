#import "settings_form_darwin.h"

// Spacing follows Apple's own settings windows, such as Safari's.
static const CGFloat OLColumnSpacing = 8;
static const CGFloat OLRowSpacing = 12;
static const CGFloat OLDescriptionSpacing = 5;
static const CGFloat OLGroupSpacing = 26;

// Views with this identifier stretch across the control column.
static NSUserInterfaceItemIdentifier const OLFillWidthIdentifier = @"OLFillWidth";

// OLLeadingCheckbox returns the checkbox at the start of view, if any.
static NSButton *_Nullable OLLeadingCheckbox(NSView *view) {
    if ([view isKindOfClass:[NSStackView class]]) {
        NSArray<NSView *> *views = ((NSStackView *)view).arrangedSubviews;
        return views.count > 0 ? OLLeadingCheckbox(views.firstObject) : nil;
    }
    if (![view isKindOfClass:[NSButton class]]) {
        return nil;
    }
    static NSButton *prototype;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        prototype = [NSButton checkboxWithTitle:@"" target:nil action:nil];
    });
    NSButton *button = (NSButton *)view;
    BOOL checkbox = button.bezelStyle == prototype.bezelStyle &&
        button.imagePosition == prototype.imagePosition &&
        button.isBordered == prototype.isBordered;
    return checkbox ? button : nil;
}

// OLTitleIndent returns how far a checkbox's title is from its leading edge,
// so descriptions under it line up with the title.
static CGFloat OLTitleIndent(NSView *view) {
    NSButton *checkbox = OLLeadingCheckbox(view);
    if (checkbox == nil) {
        return 0;
    }
    NSSize size = checkbox.fittingSize;
    NSRect title = [checkbox.cell titleRectForBounds:NSMakeRect(0, 0, size.width, size.height)];
    return NSMinX(title) > 0 ? NSMinX(title) : 20;
}

static NSTextField *OLFormLabel(NSString *text) {
    NSTextField *label = [NSTextField labelWithString:text];
    label.alignment = NSTextAlignmentRight;
    label.translatesAutoresizingMaskIntoConstraints = NO;
    return label;
}

@interface OLSettingsForm ()
@property(nonatomic, readwrite, strong) NSGridView *gridView;
@property(nonatomic, readwrite) CGFloat controlWidth;
@property(nonatomic) CGFloat descriptionIndent;
@property(nonatomic) CGFloat nextRowSpacing;
@end

@implementation OLSettingsForm

- (instancetype)initWithWidth:(CGFloat)width labelWidth:(CGFloat)labelWidth {
    self = [super init];
    if (self) {
        _controlWidth = width - labelWidth - OLColumnSpacing;
        _gridView = [NSGridView gridViewWithNumberOfColumns:2 rows:0];
        _gridView.translatesAutoresizingMaskIntoConstraints = NO;
        _gridView.columnSpacing = OLColumnSpacing;
        _gridView.rowSpacing = 0;
        _gridView.rowAlignment = NSGridRowAlignmentFirstBaseline;

        NSGridColumn *labels = [_gridView columnAtIndex:0];
        labels.xPlacement = NSGridCellPlacementTrailing;
        labels.width = labelWidth;
        NSGridColumn *controls = [_gridView columnAtIndex:1];
        controls.xPlacement = NSGridCellPlacementLeading;
        controls.width = _controlWidth;
    }
    return self;
}

- (NSGridRow *)addRowWithLabel:(NSString *)label view:(NSView *)view {
    NSView *labelView = label != nil ? OLFormLabel(label) : [NSGridCell emptyContentView];
    NSGridRow *row = [self.gridView addRowWithViews:@[labelView, view]];
    row.topPadding = self.gridView.numberOfRows == 1 ? 0 : MAX(self.nextRowSpacing, OLRowSpacing);
    if ([view.identifier isEqualToString:OLFillWidthIdentifier]) {
        [row cellAtIndex:1].xPlacement = NSGridCellPlacementFill;
    }
    self.nextRowSpacing = 0;
    self.descriptionIndent = OLTitleIndent(view);
    return row;
}

- (NSTextField *)addDescription:(NSString *)text {
    NSTextField *description = OLDescriptionLabel(text);
    description.preferredMaxLayoutWidth = self.controlWidth - self.descriptionIndent;
    [self addDescriptionView:description];
    return description;
}

- (NSGridRow *)addDescriptionView:(NSView *)view {
    CGFloat indent = self.descriptionIndent;
    NSView *content = view;
    if (indent > 0) {
        content = [[NSView alloc] initWithFrame:NSZeroRect];
        content.translatesAutoresizingMaskIntoConstraints = NO;
        [content addSubview:view];
        [NSLayoutConstraint activateConstraints:@[
            [view.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:indent],
            [view.trailingAnchor constraintLessThanOrEqualToAnchor:content.trailingAnchor],
            [view.topAnchor constraintEqualToAnchor:content.topAnchor],
            [view.bottomAnchor constraintEqualToAnchor:content.bottomAnchor],
        ]];
    }

    NSGridRow *row = [self.gridView addRowWithViews:@[[NSGridCell emptyContentView], content]];
    row.rowAlignment = NSGridRowAlignmentNone;
    row.topPadding = OLDescriptionSpacing;
    return row;
}

- (void)addGroupSpacing {
    self.nextRowSpacing = OLGroupSpacing;
}

- (void)addSeparator {
    NSBox *separator = [[NSBox alloc] initWithFrame:NSZeroRect];
    separator.boxType = NSBoxSeparator;
    separator.translatesAutoresizingMaskIntoConstraints = NO;
    NSGridRow *row = [self.gridView addRowWithViews:@[separator, [NSGridCell emptyContentView]]];
    [row mergeCellsInRange:NSMakeRange(0, 2)];
    [row cellAtIndex:0].xPlacement = NSGridCellPlacementFill;
    row.rowAlignment = NSGridRowAlignmentNone;
    row.topPadding = self.gridView.numberOfRows == 1 ? 0 : OLGroupSpacing;
    self.nextRowSpacing = OLGroupSpacing;
}

@end

NSButton *OLCheckbox(NSString *title, id target, SEL action) {
    NSButton *checkbox = [NSButton checkboxWithTitle:title target:target action:action];
    checkbox.translatesAutoresizingMaskIntoConstraints = NO;
    return checkbox;
}

NSButton *OLPushButton(NSString *title, id target, SEL action) {
    NSButton *button = [NSButton buttonWithTitle:title target:target action:action];
    button.translatesAutoresizingMaskIntoConstraints = NO;
    return button;
}

NSTextField *OLDescriptionLabel(NSString *text) {
    NSTextField *label = [NSTextField wrappingLabelWithString:text];
    label.font = [NSFont systemFontOfSize:[NSFont smallSystemFontSize]];
    label.textColor = [NSColor secondaryLabelColor];
    label.selectable = NO;
    label.translatesAutoresizingMaskIntoConstraints = NO;
    return label;
}

NSView *OLControlWithAccessory(NSView *control, NSView *accessory) {
    NSStackView *stack = [[NSStackView alloc] initWithFrame:NSZeroRect];
    stack.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    stack.alignment = NSLayoutAttributeFirstBaseline;
    stack.spacing = 8;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    stack.identifier = OLFillWidthIdentifier;
    [stack addView:control inGravity:NSStackViewGravityLeading];
    [stack addView:accessory inGravity:NSStackViewGravityTrailing];
    return stack;
}

NSStackView *OLHorizontalStack(NSArray<NSView *> *views) {
    NSStackView *stack = [NSStackView stackViewWithViews:views];
    stack.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    stack.alignment = NSLayoutAttributeFirstBaseline;
    stack.spacing = 8;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    return stack;
}
