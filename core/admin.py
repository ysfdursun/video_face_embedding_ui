from django.contrib import admin
from django.utils.html import format_html
from .models import Actor, Movie, MovieCast, FaceRecognitionSettings, FaceGroup, FaceDetection

# ========================================================
# INLINES
# ========================================================

class MovieCastInline(admin.TabularInline):
    model = MovieCast
    extra = 1
    autocomplete_fields = ['actor', 'movie']
    verbose_name = "Cast Member"
    verbose_name_plural = "Cast Members"

class FaceDetectionInline(admin.TabularInline):
    model = FaceDetection
    extra = 0
    fields = ('id', 'quality_score', 'blur_score', 'is_valid', 'is_outlier')
    readonly_fields = ('id', 'quality_score', 'blur_score', 'is_valid', 'is_outlier')
    show_change_link = True
    can_delete = False

# ========================================================
# ADMIN CONFIGURATIONS
# ========================================================

@admin.register(Actor)
class ActorAdmin(admin.ModelAdmin):
    list_display = ('name', 'cast_count')
    search_fields = ('name',)
    ordering = ('name',)
    list_per_page = 50

    def cast_count(self, obj):
        return obj.moviecast_set.count()
    cast_count.short_description = 'Movies'

@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ('title', 'cast_count', 'group_count')
    search_fields = ('title',)
    ordering = ('title',)
    inlines = [MovieCastInline]
    list_per_page = 50

    def cast_count(self, obj):
        return obj.cast_members.count()
    cast_count.short_description = 'Cast Size'

    def group_count(self, obj):
        return obj.face_groups.count()
    group_count.short_description = 'Face Groups'

@admin.register(MovieCast)
class MovieCastAdmin(admin.ModelAdmin):
    list_display = ('movie', 'actor')
    search_fields = ('movie__title', 'actor__name')
    autocomplete_fields = ['movie', 'actor']
    list_filter = ('movie',)

@admin.register(FaceGroup)
class FaceGroupAdmin(admin.ModelAdmin):
    list_display = ('group_id', 'movie_link', 'face_count', 'avg_confidence_fmt', 'risk_level_badge', 'is_labeled')
    list_filter = ('risk_level', 'is_labeled', 'movie')
    search_fields = ('group_id', 'movie__title', 'name')
    readonly_fields = ('created_at', 'updated_at', 'representative_embedding')
    inlines = [FaceDetectionInline]
    actions = ['mark_as_low_risk', 'reset_label']
    
    fieldsets = (
        ('Group Info', {
            'fields': ('movie', 'group_id', 'name', 'is_labeled')
        }),
        ('Statistics', {
            'fields': ('face_count', 'total_faces', 'avg_confidence', 'avg_quality', 'risk_level')
        }),
        ('Advanced', {
            'fields': ('representative_embedding', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def movie_link(self, obj):
        return obj.movie.title
    movie_link.short_description = 'Movie'
    movie_link.admin_order_field = 'movie'

    def avg_confidence_fmt(self, obj):
        return f"{obj.avg_confidence:.2f}"
    avg_confidence_fmt.short_description = 'Confidence'
    avg_confidence_fmt.admin_order_field = 'avg_confidence'

    def risk_level_badge(self, obj):
        colors = {
            'LOW': 'green',
            'MEDIUM': 'orange',
            'HIGH': 'red',
        }
        color = colors.get(obj.risk_level, 'black')
        return format_html(
            '<span style="color: white; background-color: {}; padding: 3px 10px; border-radius: 5px; font-weight: bold;">{}</span>',
            color,
            obj.get_risk_level_display()
        )
    risk_level_badge.short_description = 'Risk Level'
    risk_level_badge.allow_tags = True

    @admin.action(description='Mark selected as Low Risk')
    def mark_as_low_risk(self, request, queryset):
        updated = queryset.update(risk_level='LOW')
        self.message_user(request, f"{updated} groups marked as Low Risk.")

    @admin.action(description='Reset Label Status')
    def reset_label(self, request, queryset):
        updated = queryset.update(is_labeled=False, name='')
        self.message_user(request, f"{updated} groups reset.")

@admin.register(FaceDetection)
class FaceDetectionAdmin(admin.ModelAdmin):
    list_display = ('id', 'face_group_link', 'quality_score', 'blur_score', 'status_badge')
    list_filter = ('is_valid', 'is_outlier', 'face_group__movie')
    readonly_fields = ('embedding', 'created_at')
    search_fields = ('source_image', 'face_group__group_id')
    
    fieldsets = (
        ('Image Source', {
            'fields': ('source_image', 'frame_number', 'bbox', 'face_group')
        }),
        ('Quality Metrics', {
            'fields': ('quality_score', 'blur_score', 'illumination_score', 'pose_score', 'detection_confidence')
        }),
        ('Status', {
            'fields': ('is_valid', 'is_outlier', 'image_hash')
        }),
        ('Data', {
            'fields': ('embedding', 'created_at'),
            'classes': ('collapse',)
        }),
    )

    def face_group_link(self, obj):
        if obj.face_group:
            return f"{obj.face_group.group_id} ({obj.face_group.movie.title})"
        return "-"
    face_group_link.short_description = 'Group'

    def status_badge(self, obj):
        if obj.is_outlier:
            return format_html('<span style="color:red; font-weight:bold;">OUTLIER</span>')
        if not obj.is_valid:
            return format_html('<span style="color:gray;">INVALID</span>')
        return format_html('<span style="color:green;">VALID</span>')
    status_badge.short_description = 'Status'

@admin.register(FaceRecognitionSettings)
class FaceRecognitionSettingsAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'detection_threshold', 'min_face_size', 'gpu_enabled')
    
    fieldsets = (
        ('Detection Configuration', {
            'fields': ('detection_threshold', 'min_face_size', 'gpu_enabled'),
            'description': 'Core settings for the face detector model.'
        }),
        ('Grouping & Recognition', {
            'fields': ('grouping_threshold', 'redundancy_threshold', 'recognition_threshold_override'),
            'description': 'Parameters controlling how faces are clustered and matched. (Note: recognition_threshold usually in config.py)'
        }),
        ('Quality Filters', {
            'fields': ('quality_score_threshold', 'blur_threshold', 'pose_threshold'),
            'description': 'Minimum quality standards to accept a face.'
        }),
        ('Performance', {
            'fields': ('frame_skip_extract', 'frame_skip_group'),
            'classes': ('collapse',),
        }),
    )
    
    # Dynamic field retrieval to avoid errors if I guessed field names wrong compared to models.py
    # Re-checking models.py content from viewing history:
    # Fields: detection_threshold, grouping_threshold, min_face_size, frame_skip_extract, frame_skip_group, gpu_enabled, quality_threshold, redundancy_threshold.
    
    # Corrected Fieldsets based on actual models.py:
    fieldsets = (
        ('Detection', {
            'fields': ('detection_threshold', 'min_face_size', 'gpu_enabled')
        }),
        ('Grouping & Logic', {
            'fields': ('grouping_threshold', 'redundancy_threshold')
        }),
        ('Quality Control', {
            'fields': ('quality_threshold',) 
        }),
         ('Performance', {
            'fields': ('frame_skip_extract', 'frame_skip_group'),
             'classes': ('collapse',)
        }),
    )

    def has_add_permission(self, request):
        # Singleton pattern: Only allow add if none exists
        if self.model.objects.exists():
            return False
        return True
