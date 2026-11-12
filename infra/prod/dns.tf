# Delegated subdomain: the four name_servers are NS records for host "paperlens"
# in the ailearn-vaic.cloud zone at Mat Bao. The apex stays at Mat Bao.
#
# A reusable delegation set pins the nameservers, so recreating the zone never
# requires editing the records at Mat Bao again.
import {
  to = aws_route53_delegation_set.main
  id = "N0313081B7E1Q3SEKLKU"
}

resource "aws_route53_delegation_set" "main" {
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_route53_zone" "main" {
  name              = var.domain
  delegation_set_id = aws_route53_delegation_set.main.id
}

resource "aws_acm_certificate" "api" {
  domain_name       = local.api_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "api_cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.api.domain_validation_options : dvo.domain_name => dvo
  }
  zone_id         = aws_route53_zone.main.zone_id
  name            = each.value.resource_record_name
  type            = each.value.resource_record_type
  records         = [each.value.resource_record_value]
  ttl             = 300
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "api" {
  certificate_arn         = aws_acm_certificate.api.arn
  validation_record_fqdns = [for record in aws_route53_record.api_cert_validation : record.fqdn]
}

resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = local.api_domain
  type    = "A"
  alias {
    name                   = aws_lb.api.dns_name
    zone_id                = aws_lb.api.zone_id
    evaluate_target_health = true
  }
}
